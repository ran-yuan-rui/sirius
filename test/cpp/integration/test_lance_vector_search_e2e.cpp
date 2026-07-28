/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scan/lance_test_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/appender.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <op/sirius_physical_delim_join.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs    = std::filesystem;
namespace lance = sirius::lance;
namespace test  = sirius::test::lance;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

std::string lance_call(std::string query = "[0.1, 0.2, 0.3, 0.4]::FLOAT[]",
                       std::string named = ", k := 4")
{
  if (query.empty()) { query = "[0.1, 0.2, 0.3, 0.4]::FLOAT[]"; }
  return "sirius_lance_vector_search("
         "'memory://fake.lance', 'embedding', " +
         std::move(query) + std::move(named) + ")";
}

test::dataset_script script_with_events(std::vector<test::stream_event> events,
                                        std::size_t rows         = 4,
                                        std::size_t vector_width = 4,
                                        std::int64_t id_base     = 0)
{
  test::dataset_script script;
  script.schema    = test::make_knn_batch(0, vector_width, id_base);
  script.events    = std::move(events);
  script.row_count = static_cast<std::int64_t>(rows);
  return script;
}

std::shared_ptr<test::scripted_lance_ffi> one_batch_api(std::size_t rows     = 4,
                                                        std::int64_t id_base = 0)
{
  return std::make_shared<test::scripted_lance_ffi>(std::vector{
    script_with_events({test::stream_event::make_batch(test::make_knn_batch(rows, 4, id_base)),
                        test::stream_event::make_eos()},
                       rows,
                       4,
                       id_base)});
}

struct lance_sql_fixture {
  lance_sql_fixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    REQUIRE(sirius::test::g_integration_env->is_active());
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->database());
    require_ok("SET gpu_execution = true");
    require_ok("SET enable_duckdb_fallback = true");
  }

  void require_ok(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  }

  std::string require_error(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    REQUIRE(result->HasError());
    return result->GetError();
  }

  std::unique_ptr<duckdb::Connection> con;
};

duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(duckdb::Connection& con,
                                                                  std::string const& sql)
{
  auto& context          = *con.context;
  auto original_disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius_physical_operator> result;
  try {
    duckdb::Parser parser(context.GetParserOptions());
    parser.ParseQuery(sql);
    REQUIRE(parser.statements.size() == 1);

    duckdb::Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    auto plan = std::move(planner.plan);
    if (context.config.enable_optimizer) {
      duckdb::Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver::Verify(*plan);
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator generator(context);
    result = generator.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

template <typename Fn>
void visit_plan(sirius_physical_operator* root, Fn const& fn)
{
  if (!root) { return; }
  fn(*root);
  for (auto& child : root->children) {
    visit_plan(child.get(), fn);
  }
  if (root->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    visit_plan(delim.join.get(), fn);
    visit_plan(delim.distinct_root.get(), fn);
  }
}

std::size_t count_operators(sirius_physical_operator* root, SiriusPhysicalOperatorType type)
{
  std::size_t count = 0;
  visit_plan(root, [&](sirius_physical_operator const& op) {
    if (op.type == type) { ++count; }
  });
  return count;
}

class temp_directory {
 public:
  explicit temp_directory(std::string name)
  {
    static std::atomic<std::uint64_t> ordinal{0};
    _path = fs::temp_directory_path() / (std::move(name) + "-" + std::to_string(::getpid()) + "-" +
                                         std::to_string(ordinal.fetch_add(1)));
    std::error_code ec;
    fs::remove_all(_path, ec);
    fs::create_directories(_path);
  }

  ~temp_directory()
  {
    std::error_code ec;
    fs::remove_all(_path, ec);
  }

  fs::path const& path() const { return _path; }

 private:
  fs::path _path;
};

void write_docs_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = con.Query(
    "COPY (SELECT i::BIGINT AS doc_id, "
    "CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END AS category "
    "FROM range(6) t(i)) TO '" +
    path.string() + "' (FORMAT PARQUET)");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
}

constexpr std::size_t kAc5Rows = 2000;
constexpr std::size_t kAc5Dim  = 8;
constexpr std::size_t kAc5K    = 10;
constexpr float kAc5QueryValue = 0.50025F;

char const* kAc5QueryVectorSql =
  "[0.50025, 0.50025, 0.50025, 0.50025, "
  "0.50025, 0.50025, 0.50025, 0.50025]::FLOAT[]";

std::string ac5_lance_call(fs::path const& dataset,
                           std::size_t k,
                           bool use_index,
                           std::string extra_options = {})
{
  return "sirius_lance_vector_search('" + dataset.string() + "', 'vec', " + kAc5QueryVectorSql +
         ", k := " + std::to_string(k) + ", use_index := " + (use_index ? "true" : "false") +
         std::move(extra_options) + ")";
}

void write_ac5_lance_fixture(fs::path const& dataset)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  duckdb::DuckDB db(nullptr);
  duckdb::Connection writer(db);

  auto require_writer_ok = [](auto const& result) {
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  };

  auto loaded = writer.Query("LOAD lance");
  require_writer_ok(loaded);

  auto copied = writer.Query(
    "COPY (SELECT i::BIGINT AS doc_id, (i % 7)::INTEGER AS cat, "
    "(i::DOUBLE / 1000.0) AS score, "
    "list_transform(range(8), lambda x: (i::FLOAT / 1000.0))::FLOAT[8] AS vec "
    "FROM range(2000) t(i)) TO '" +
    dataset.string() + "' (FORMAT lance, mode 'overwrite')");
  require_writer_ok(copied);

  auto indexed = writer.Query("CREATE INDEX vec_idx ON '" + dataset.string() +
                              "' (vec) USING IVF_PQ WITH "
                              "(num_partitions=8, num_sub_vectors=4, metric_type='l2')");
  require_writer_ok(indexed);

  auto indexes = writer.Query("SHOW INDEXES ON '" + dataset.string() + "'");
  require_writer_ok(indexes);
  REQUIRE(indexes->RowCount() == 1);
  CHECK(indexes->GetValue(0, 0).ToString() == "vec_idx");
  CHECK(indexes->GetValue(1, 0).ToString() == "IVF_PQ");
  CHECK(indexes->GetValue(3, 0).GetValue<std::uint64_t>() == kAc5Rows);
}

struct expected_neighbor {
  std::int64_t doc_id;
  double distance;
};

std::vector<expected_neighbor> ac5_exact_oracle()
{
  std::vector<expected_neighbor> result;
  result.reserve(kAc5Rows);
  for (std::size_t i = 0; i < kAc5Rows; ++i) {
    auto const value = static_cast<float>(i) / 1000.0F;
    auto const delta = value - kAc5QueryValue;
    result.push_back(
      {static_cast<std::int64_t>(i),
       static_cast<double>(kAc5Dim) * static_cast<double>(delta) * static_cast<double>(delta)});
  }
  std::sort(result.begin(), result.end(), [](auto const& left, auto const& right) {
    if (left.distance != right.distance) { return left.distance < right.distance; }
    return left.doc_id < right.doc_id;
  });
  result.resize(kAc5K);
  return result;
}

struct copy_failure_state {
  enum class error_kind { NOT_IMPLEMENTED, INTERNAL, INTERRUPT };

  std::atomic<std::uint64_t> bind_calls{0};
  std::atomic<std::uint64_t> cpu_calls{0};
  error_kind kind{error_kind::NOT_IMPLEMENTED};
};

std::shared_ptr<copy_failure_state>& active_copy_failure_state()
{
  static std::shared_ptr<copy_failure_state> state;
  return state;
}

class scoped_copy_failure_state {
 public:
  explicit scoped_copy_failure_state(std::shared_ptr<copy_failure_state> state)
  {
    active_copy_failure_state() = std::move(state);
  }

  ~scoped_copy_failure_state() { active_copy_failure_state().reset(); }

  scoped_copy_failure_state(scoped_copy_failure_state const&)            = delete;
  scoped_copy_failure_state& operator=(scoped_copy_failure_state const&) = delete;
};

struct noncopyable_lance_bind_data : duckdb::TableFunctionData {
  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    throw duckdb::NotImplementedException("forced Lance bind-data copy failure");
  }

  bool Equals(duckdb::FunctionData const&) const override { return true; }
};

duckdb::unique_ptr<duckdb::FunctionData> copy_failure_bind(
  duckdb::ClientContext&,
  duckdb::TableFunctionBindInput&,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<std::string>& names)
{
  auto state = active_copy_failure_state();
  if (!state) { throw duckdb::InternalException("copy-failure test state is not installed"); }
  auto const call = state->bind_calls.fetch_add(1);
  if (call > 0) {
    switch (state->kind) {
      case copy_failure_state::error_kind::NOT_IMPLEMENTED:
        throw duckdb::NotImplementedException("forced Lance replan failure");
      case copy_failure_state::error_kind::INTERNAL:
        throw duckdb::InternalException("forced Lance replan internal failure");
      case copy_failure_state::error_kind::INTERRUPT: throw duckdb::InterruptException();
    }
  }
  return_types.push_back(duckdb::LogicalType::BIGINT);
  names.push_back("doc_id");
  return duckdb::make_uniq<noncopyable_lance_bind_data>();
}

void copy_failure_execute(duckdb::ClientContext&,
                          duckdb::TableFunctionInput&,
                          duckdb::DataChunk& output)
{
  auto state = active_copy_failure_state();
  if (state) { state->cpu_calls.fetch_add(1); }
  output.SetCardinality(0);
}

void register_copy_failure_overload(duckdb::DuckDB& db)
{
  static std::once_flag once;
  std::call_once(once, [&db] {
    duckdb::TableFunction function("sirius_lance_vector_search",
                                   {duckdb::LogicalType::BIGINT},
                                   copy_failure_execute,
                                   copy_failure_bind);
    function.projection_pushdown = true;
    duckdb::CreateTableFunctionInfo info(std::move(function));
    info.on_conflict = duckdb::OnCreateConflict::ALTER_ON_CONFLICT;
    auto& catalog    = duckdb::Catalog::GetSystemCatalog(*db.instance);
    auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(*db.instance);
    catalog.CreateFunction(transaction, info);
  });
}

}  // namespace

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance SQL bind preserves open errors and validates scalar formats",
                 "[lance][integration][bind][schema]")
{
  SECTION("dataset-open error creates no handles")
  {
    auto script       = script_with_events({});
    script.open_error = lance::lance_error{11, "fake dataset open denied"};
    auto api          = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});
    lance::scoped_ffi_api_override override([api] { return api; });

    auto error = require_error("SELECT doc_id FROM " + lance_call());
    CHECK_THAT(error, Catch::Contains("fake dataset open denied"));
    CHECK(api->counters()->dataset_opens.load() == 1);
    CHECK(api->counters()->dataset_closes.load() == 0);
    CHECK(api->counters()->stream_opens.load() == 0);
  }

  SECTION("large utf8 is legal")
  {
    auto script                 = script_with_events({});
    script.schema.mutate_schema = [](ArrowSchema& schema) { schema.children[1]->format = "U"; };
    auto api                    = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});
    lance::scoped_ffi_api_override override([api] { return api; });

    auto prepared = con->Prepare("SELECT label FROM " + lance_call());
    REQUIRE(prepared);
    if (prepared->HasError()) { UNSCOPED_INFO(prepared->GetError()); }
    REQUIRE_FALSE(prepared->HasError());
    CHECK(api->counters()->schema_releases.load() == 1);
    CHECK(api->counters()->dataset_closes.load() == 1);
  }

  SECTION("decimal is rejected with its column and format")
  {
    auto script                 = script_with_events({});
    script.schema.mutate_schema = [](ArrowSchema& schema) {
      schema.children[1]->name   = "price";
      schema.children[1]->format = "d:10,2";
    };
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});
    lance::scoped_ffi_api_override override([api] { return api; });

    auto error = require_error("SELECT * FROM " + lance_call());
    CHECK_THAT(error, Catch::Contains("price"));
    CHECK_THAT(error, Catch::Contains("d:10,2"));
    CHECK(api->counters()->schema_releases.load() == 1);
    CHECK(api->counters()->dataset_closes.load() == 1);
  }

  SECTION("non-microsecond timestamp is rejected")
  {
    auto script                 = script_with_events({});
    script.schema.mutate_schema = [](ArrowSchema& schema) {
      schema.children[1]->name   = "created_at";
      schema.children[1]->format = "tsn:";
    };
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});
    lance::scoped_ffi_api_override override([api] { return api; });

    auto error = require_error("SELECT * FROM " + lance_call());
    CHECK_THAT(error, Catch::Contains("created_at"));
    CHECK_THAT(error, Catch::Contains("tsn:"));
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance SQL bind validates vector identity format and query width",
                 "[lance][integration][bind][vector]")
{
  SECTION("vector column is missing")
  {
    auto script                 = script_with_events({});
    script.schema.mutate_schema = [](ArrowSchema& schema) {
      schema.children[2]->name = "other_vector";
    };
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});
    lance::scoped_ffi_api_override override([api] { return api; });

    auto error = require_error("SELECT doc_id FROM " + lance_call());
    CHECK_THAT(error, Catch::Contains("embedding"));
    CHECK_THAT(error, Catch::Contains("not found"));
  }

  SECTION("vector column is not fixed size")
  {
    auto script                 = script_with_events({});
    script.schema.mutate_schema = [](ArrowSchema& schema) { schema.children[2]->format = "+l"; };
    auto api                    = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});
    lance::scoped_ffi_api_override override([api] { return api; });

    auto error = require_error("SELECT doc_id FROM " + lance_call());
    CHECK_THAT(error, Catch::Contains("embedding"));
    CHECK_THAT(error, Catch::Contains("+w:"));
  }

  SECTION("query dimension differs from vector width")
  {
    auto api = one_batch_api();
    lance::scoped_ffi_api_override override([api] { return api; });
    auto error = require_error("SELECT doc_id FROM " + lance_call("[0.1, 0.2, 0.3]::FLOAT[]"));
    CHECK_THAT(error, Catch::Contains("dimension"));
    CHECK_THAT(error, Catch::Contains("4"));
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance SQL bind rejects invalid k secrets and the runtime kill switch",
                 "[lance][integration][bind][options]")
{
  SECTION("k zero is a bind error")
  {
    auto api = one_batch_api();
    lance::scoped_ffi_api_override override([api] { return api; });
    auto error = require_error("SELECT doc_id FROM " + lance_call({}, ", k := 0"));
    CHECK_THAT(error, Catch::Contains("k"));
    CHECK_THAT(error, Catch::Contains("positive"));
    CHECK(api->counters()->dataset_opens.load() == 0);
  }

  SECTION("k above the configured maximum is a bind error")
  {
    auto api = one_batch_api();
    lance::scoped_ffi_api_override override([api] { return api; });
    auto error = require_error("SELECT doc_id FROM " + lance_call({}, ", k := 1000001"));
    CHECK_THAT(error, Catch::Contains("sirius_lance_max_k"));
    CHECK(api->counters()->dataset_opens.load() == 0);
  }

  SECTION("credential-bearing storage option is rejected")
  {
    auto api = one_batch_api();
    lance::scoped_ffi_api_override override([api] { return api; });
    auto error = require_error(
      "SELECT doc_id FROM " +
      lance_call({}, ", k := 4, storage_options := MAP {'access_key_id': 'must-not-enter-sql'}"));
    CHECK_THAT(error, Catch::Contains("access_key_id"));
    CHECK_THAT(error, Catch::Contains("credential"));
    CHECK(api->counters()->dataset_opens.load() == 0);
  }

  SECTION("kill switch rejects before opening the dataset")
  {
    auto api = one_batch_api();
    lance::scoped_ffi_api_override override([api] { return api; });
    require_ok("SET sirius_lance_knn_enabled = false");
    auto error = require_error("SELECT doc_id FROM " + lance_call());
    CHECK_THAT(error, Catch::Contains("sirius_lance_knn_enabled"));
    CHECK(api->counters()->dataset_opens.load() == 0);
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance GPU source returns exact fake KNN rows across edge batch shapes",
                 "[lance][integration][correctness]")
{
  SECTION("single nullable batch")
  {
    auto api = one_batch_api(3);
    lance::scoped_ffi_api_override override([api] { return api; });
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query("SELECT doc_id, label, _distance FROM " + lance_call({}, ", k := 3") +
                             " ORDER BY doc_id");
    auto after  = sirius::test::get_transparent_execution_stats(*con);

    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->RowCount() == 3);
    CHECK(result->GetValue(0, 0).GetValue<std::int64_t>() == 0);
    CHECK(result->GetValue(0, 2).GetValue<std::int64_t>() == 2);
    CHECK(result->GetValue(1, 1).IsNull());
    CHECK(result->GetValue(2, 2).GetValue<float>() == Approx(0.2F));
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);
  }

  SECTION("multiple batches feed one aggregate")
  {
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector{script_with_events({test::stream_event::make_batch(test::make_knn_batch(2, 4, 0)),
                                      test::stream_event::make_batch(test::make_knn_batch(3, 4, 2)),
                                      test::stream_event::make_eos()},
                                     5)});
    lance::scoped_ffi_api_override override([api] { return api; });
    auto result =
      con->Query("SELECT count(*), min(doc_id), max(doc_id) FROM " + lance_call({}, ", k := 5"));
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    CHECK(result->GetValue(0, 0).GetValue<std::int64_t>() == 5);
    CHECK(result->GetValue(1, 0).GetValue<std::int64_t>() == 0);
    CHECK(result->GetValue(2, 0).GetValue<std::int64_t>() == 4);
  }

  SECTION("empty result still completes")
  {
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector{script_with_events({test::stream_event::make_eos()}, 0)});
    lance::scoped_ffi_api_override override([api] { return api; });
    auto result = con->Query("SELECT count(*) FROM " + lance_call());
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    CHECK(result->GetValue(0, 0).GetValue<std::int64_t>() == 0);
  }

  SECTION("k above row count is not an error and returns at most k rows")
  {
    auto api = one_batch_api(2);
    lance::scoped_ffi_api_override override([api] { return api; });
    auto result = con->Query("SELECT doc_id FROM " + lance_call({}, ", k := 10"));
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    CHECK(result->RowCount() <= 10);
    CHECK(result->RowCount() == 2);
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance and Parquet form two GPU scans with GPU relational operators",
                 "[lance][integration][planner][join]")
{
  temp_directory tmp("sirius-lance-parquet");
  auto parquet = tmp.path() / "docs.parquet";
  write_docs_parquet(parquet);

  auto api = one_batch_api(6);
  lance::scoped_ffi_api_override override([api] { return api; });
  auto sql = "WITH candidates AS (SELECT doc_id, _distance FROM " + lance_call({}, ", k := 6") +
             ") SELECT p.category, count(*) AS n, avg(c._distance) AS ad "
             "FROM candidates c JOIN read_parquet('" +
             parquet.string() +
             "') p ON c.doc_id = p.doc_id "
             "WHERE c._distance >= 0.1 GROUP BY p.category ORDER BY p.category";

  auto plan = generate_sirius_plan(*con, sql);
  REQUIRE(plan);
  CHECK(count_operators(plan.get(), SiriusPhysicalOperatorType::GPU_SCAN) == 2);
  CHECK(count_operators(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN) == 1);
  CHECK(count_operators(plan.get(), SiriusPhysicalOperatorType::FILTER) >= 1);
  CHECK(count_operators(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY) +
          count_operators(plan.get(), SiriusPhysicalOperatorType::PERFECT_HASH_GROUP_BY) >=
        1);

  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query(sql);
  auto after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 2);
  CHECK(result->GetValue(0, 0).ToString() == "even");
  CHECK(result->GetValue(1, 0).GetValue<std::int64_t>() == 2);
  CHECK(result->GetValue(2, 0).GetValue<double>() == Approx(0.3));
  CHECK(result->GetValue(0, 1).ToString() == "odd");
  CHECK(result->GetValue(1, 1).GetValue<std::int64_t>() == 3);
  CHECK(result->GetValue(2, 1).GetValue<double>() == Approx(0.3));
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance prepared execution reopens the dataset and observes a new snapshot",
                 "[lance][integration][prepared]")
{
  auto bind          = script_with_events({}, 1);
  auto first_rebind  = script_with_events({}, 1);
  auto first_execute = script_with_events(
    {test::stream_event::make_batch(test::make_knn_batch(1, 4, 5)), test::stream_event::make_eos()},
    1,
    4,
    5);
  auto second_rebind  = script_with_events({}, 1);
  auto second_execute = script_with_events(
    {test::stream_event::make_batch(test::make_knn_batch(1, 4, 9)), test::stream_event::make_eos()},
    1,
    4,
    9);
  auto api = std::make_shared<test::scripted_lance_ffi>(
    std::vector<test::dataset_script>{std::move(bind),
                                      std::move(first_rebind),
                                      std::move(first_execute),
                                      std::move(second_rebind),
                                      std::move(second_execute)});
  lance::scoped_ffi_api_override override([api] { return api; });

  auto before   = sirius::test::get_transparent_execution_stats(*con);
  auto prepared = con->Prepare("SELECT sum(doc_id) FROM " + lance_call({}, ", k := 1"));
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());

  duckdb::vector<duckdb::Value> values;
  auto first_result = prepared->Execute(values, /*allow_stream_result=*/false);
  REQUIRE(first_result);
  REQUIRE_FALSE(first_result->HasError());
  auto first_chunk = first_result->Fetch();
  REQUIRE(first_chunk);
  REQUIRE(first_chunk->size() == 1);
  CHECK(first_chunk->GetValue(0, 0).GetValue<std::int64_t>() == 5);
  first_result.reset();

  auto second_result = prepared->Execute(values, /*allow_stream_result=*/false);
  REQUIRE(second_result);
  REQUIRE_FALSE(second_result->HasError());
  auto second_chunk = second_result->Fetch();
  REQUIRE(second_chunk);
  REQUIRE(second_chunk->size() == 1);
  CHECK(second_chunk->GetValue(0, 0).GetValue<std::int64_t>() == 9);

  auto after = sirius::test::get_transparent_execution_stats(*con);
  sirius::test::require_transparent_execution_delta(before, after, 3, 0, 2, 0);
  CHECK(api->counters()->dataset_opens.load() == 5);
  CHECK(api->counters()->dataset_closes.load() == 5);
  CHECK(api->counters()->stream_opens.load() == 2);
  CHECK(api->counters()->stream_closes.load() == 2);
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance runtime failure preserves its error and forbids CPU replay",
                 "[lance][integration][no_fallback][runtime]")
{
  auto bind = script_with_events({}, 4);
  auto execute =
    script_with_events({test::stream_event::make_error(77, "fake Lance runtime failure")}, 4);
  auto api = std::make_shared<test::scripted_lance_ffi>(
    std::vector<test::dataset_script>{std::move(bind), std::move(execute)});
  lance::scoped_ffi_api_override override([api] { return api; });

  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query("SELECT sum(doc_id) FROM " + lance_call());
  auto after  = sirius::test::get_transparent_execution_stats(*con);

  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK_THAT(result->GetError(), Catch::Contains("fake Lance runtime failure"));
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);
  CHECK(api->counters()->stream_next_calls.load() == 1);
  CHECK(api->counters()->stream_closes.load() == 1);
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance planner rejection preserves diagnostics and forbids CPU replay",
                 "[lance][integration][no_fallback][planner]")
{
  auto api = one_batch_api();
  lance::scoped_ffi_api_override override([api] { return api; });

  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query("SELECT rowid FROM " + lance_call());
  auto after  = sirius::test::get_transparent_execution_stats(*con);

  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK_THAT(result->GetError(), Catch::Contains("rowid"));
  CHECK_THAT(result->GetError(), Catch::Contains("GPU-only"));
  sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0, 0);
  CHECK(api->counters()->stream_opens.load() == 0);
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance copy-failure replan path retains no-fallback capability",
                 "[lance][integration][no_fallback][copy]")
{
  auto state = std::make_shared<copy_failure_state>();
  scoped_copy_failure_state state_scope(state);
  register_copy_failure_overload(sirius::test::g_integration_env->database());

  SECTION("NotImplemented diagnostic is preserved")
  {
    state->kind = copy_failure_state::error_kind::NOT_IMPLEMENTED;
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query("SELECT doc_id FROM sirius_lance_vector_search(123::BIGINT)");
    auto after  = sirius::test::get_transparent_execution_stats(*con);

    REQUIRE(result);
    REQUIRE(result->HasError());
    CHECK_THAT(result->GetError(), Catch::Contains("forced Lance replan failure"));
    CHECK_THAT(result->GetError(), Catch::Contains("GPU-only"));
    sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0, 0);
    CHECK(state->cpu_calls.load() == 0);
  }

  SECTION("INTERNAL is sanitized and the session remains usable")
  {
    state->kind = copy_failure_state::error_kind::INTERNAL;
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query("SELECT doc_id FROM sirius_lance_vector_search(123::BIGINT)");
    auto after  = sirius::test::get_transparent_execution_stats(*con);

    REQUIRE(result);
    REQUIRE(result->HasError());
    CHECK_THAT(result->GetError(), Catch::Contains("forced Lance replan internal failure"));
    sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0, 0);
    CHECK(state->cpu_calls.load() == 0);

    auto healthy = con->Query("SELECT 42");
    REQUIRE(healthy);
    REQUIRE_FALSE(healthy->HasError());
  }

  SECTION("INTERRUPT remains an interrupt and is never replayed")
  {
    state->kind = copy_failure_state::error_kind::INTERRUPT;
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query("SELECT doc_id FROM sirius_lance_vector_search(123::BIGINT)");
    auto after  = sirius::test::get_transparent_execution_stats(*con);

    REQUIRE(result);
    REQUIRE(result->HasError());
    CHECK(result->GetErrorObject().Type() == duckdb::ExceptionType::INTERRUPT);
    sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0, 0);
    CHECK(state->cpu_calls.load() == 0);
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance real FFI relational tail matches a materialized CPU oracle",
                 "[lance][integration][real_ffi][oracle][relational]")
{
  temp_directory tmp("sirius-lance-ac5-relational");
  auto dataset = tmp.path() / "oracle.lance";
  write_ac5_lance_fixture(dataset);

  auto source   = ac5_lance_call(dataset, 70, false);
  auto snapshot = con->Query("SELECT doc_id, cat, score, _distance FROM " + source);
  REQUIRE(snapshot);
  if (snapshot->HasError()) { UNSCOPED_INFO(snapshot->GetError()); }
  REQUIRE_FALSE(snapshot->HasError());
  REQUIRE(snapshot->RowCount() == 70);

  require_ok("SET gpu_execution = false");
  require_ok(
    "CREATE TEMP TABLE knn_snapshot("
    "doc_id BIGINT, cat INTEGER, score DOUBLE, _distance FLOAT)");
  {
    duckdb::Appender appender(*con, "knn_snapshot");
    for (std::size_t row = 0; row < snapshot->RowCount(); ++row) {
      appender.BeginRow();
      appender.Append<std::int64_t>(snapshot->GetValue(0, row).GetValue<std::int64_t>());
      appender.Append<std::int32_t>(snapshot->GetValue(1, row).GetValue<std::int32_t>());
      appender.Append<double>(snapshot->GetValue(2, row).GetValue<double>());
      appender.Append<float>(snapshot->GetValue(3, row).GetValue<float>());
      appender.EndRow();
    }
    appender.Close();
  }
  require_ok("SET gpu_execution = true");

  auto relational_sql =
    "SELECT cat, count(*) AS n, avg(score) AS avg_score, avg(_distance) AS avg_distance FROM " +
    source +
    " WHERE score >= 0.48 AND score < 0.53 "
    "GROUP BY cat ORDER BY cat";
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto gpu    = con->Query(relational_sql);
  auto after  = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(gpu);
  if (gpu->HasError()) { UNSCOPED_INFO(gpu->GetError()); }
  REQUIRE_FALSE(gpu->HasError());
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);

  require_ok("SET gpu_execution = false");
  auto cpu = con->Query(
    "SELECT cat, count(*) AS n, avg(score) AS avg_score, avg(_distance) AS avg_distance "
    "FROM knn_snapshot WHERE score >= 0.48 AND score < 0.53 "
    "GROUP BY cat ORDER BY cat");
  REQUIRE(cpu);
  if (cpu->HasError()) { UNSCOPED_INFO(cpu->GetError()); }
  REQUIRE_FALSE(cpu->HasError());

  REQUIRE(gpu->RowCount() == cpu->RowCount());
  REQUIRE(gpu->ColumnCount() == cpu->ColumnCount());
  for (std::size_t row = 0; row < gpu->RowCount(); ++row) {
    CHECK(gpu->GetValue(0, row).ToString() == cpu->GetValue(0, row).ToString());
    CHECK(gpu->GetValue(1, row).GetValue<std::int64_t>() ==
          cpu->GetValue(1, row).GetValue<std::int64_t>());
    CHECK(gpu->GetValue(2, row).GetValue<double>() ==
          Approx(cpu->GetValue(2, row).GetValue<double>()).epsilon(1e-6));
    CHECK(gpu->GetValue(3, row).GetValue<double>() ==
          Approx(cpu->GetValue(3, row).GetValue<double>()).epsilon(1e-5).margin(1e-9));
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance real FFI exact KNN matches an independent distance oracle",
                 "[lance][integration][real_ffi][oracle][exact]")
{
  temp_directory tmp("sirius-lance-ac5-exact");
  auto dataset = tmp.path() / "oracle.lance";
  write_ac5_lance_fixture(dataset);

  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result =
    con->Query("SELECT doc_id, _distance FROM " + ac5_lance_call(dataset, kAc5K, false));
  auto after = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == kAc5K);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 0);

  auto const expected = ac5_exact_oracle();
  for (std::size_t row = 0; row < kAc5K; ++row) {
    CHECK(result->GetValue(0, row).GetValue<std::int64_t>() == expected[row].doc_id);
    CHECK(result->GetValue(1, row).GetValue<float>() ==
          Approx(expected[row].distance).epsilon(1e-4).margin(1e-9));
  }
}

TEST_CASE_METHOD(lance_sql_fixture,
                 "Lance real FFI indexed KNN meets recall against exact search",
                 "[lance][integration][real_ffi][oracle][recall]")
{
  temp_directory tmp("sirius-lance-ac5-recall");
  auto dataset = tmp.path() / "oracle.lance";
  write_ac5_lance_fixture(dataset);

  auto exact = con->Query("SELECT doc_id FROM " + ac5_lance_call(dataset, kAc5K, false));
  REQUIRE(exact);
  if (exact->HasError()) { UNSCOPED_INFO(exact->GetError()); }
  REQUIRE_FALSE(exact->HasError());
  REQUIRE(exact->RowCount() == kAc5K);

  auto indexed =
    con->Query("SELECT doc_id FROM " +
               ac5_lance_call(dataset, kAc5K, true, ", nprobes := 1, refine_factor := 1"));
  REQUIRE(indexed);
  if (indexed->HasError()) { UNSCOPED_INFO(indexed->GetError()); }
  REQUIRE_FALSE(indexed->HasError());
  REQUIRE(indexed->RowCount() == kAc5K);

  std::set<std::int64_t> exact_ids;
  std::set<std::int64_t> indexed_ids;
  for (std::size_t row = 0; row < kAc5K; ++row) {
    exact_ids.insert(exact->GetValue(0, row).GetValue<std::int64_t>());
    indexed_ids.insert(indexed->GetValue(0, row).GetValue<std::int64_t>());
  }
  REQUIRE(exact_ids.size() == kAc5K);
  REQUIRE(indexed_ids.size() == kAc5K);

  std::size_t overlap = 0;
  for (auto const id : indexed_ids) {
    if (exact_ids.count(id) != 0) { ++overlap; }
  }
  auto const recall = static_cast<double>(overlap) / static_cast<double>(kAc5K);
  UNSCOPED_INFO("Lance IVF_PQ recall@" << kAc5K << " = " << recall);
  CHECK(recall >= 0.7);
}
