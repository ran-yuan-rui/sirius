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

#include <duckdb/common/exception.hpp>
#include <duckdb/common/serializer/deserializer.hpp>
#include <duckdb/common/serializer/serializer.hpp>
#include <duckdb/common/table_column.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <lance_shim/lance_bind_data.hpp>
#include <lance_shim/lance_table_function.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::lance {

namespace {

/// Storage-option keys that carry secrets. They must never travel through SQL
/// text, where they would land in logs, error messages, and prepared plans;
/// credentials come from the ambient provider chain instead.
constexpr std::array kCredentialKeys = {
  "access_key_id",
  "secret_access_key",
  "session_token",
  "aws_access_key_id",
  "aws_secret_access_key",
  "aws_session_token",
};

/// One RAII close for the bind-time dataset handle, which exists only long
/// enough to read the KNN schema.
class scoped_dataset {
 public:
  scoped_dataset(std::shared_ptr<lance_ffi_api> api, void* handle)
    : _api(std::move(api)), _handle(handle)
  {
  }

  ~scoped_dataset()
  {
    if (_handle != nullptr) { _api->dataset_close(_handle); }
  }

  scoped_dataset(scoped_dataset const&)            = delete;
  scoped_dataset& operator=(scoped_dataset const&) = delete;

  [[nodiscard]] void* get() const noexcept { return _handle; }

 private:
  std::shared_ptr<lance_ffi_api> _api;
  void* _handle;
};

/// Same idea for the exported schema: the Arrow C contract puts the release on
/// the consumer.
class scoped_arrow_schema {
 public:
  ~scoped_arrow_schema()
  {
    if (schema.release != nullptr) { schema.release(&schema); }
  }

  scoped_arrow_schema()                                      = default;
  scoped_arrow_schema(scoped_arrow_schema const&)            = delete;
  scoped_arrow_schema& operator=(scoped_arrow_schema const&) = delete;

  ArrowSchema schema{};
};

bool setting_enabled(duckdb::ClientContext& context, char const* name, bool fallback)
{
  duckdb::Value value;
  if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) {
    return value.GetValue<bool>();
  }
  return fallback;
}

std::uint64_t setting_uint(duckdb::ClientContext& context, char const* name, std::uint64_t fallback)
{
  duckdb::Value value;
  if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) {
    auto const raw = value.GetValue<std::int64_t>();
    if (raw > 0) { return static_cast<std::uint64_t>(raw); }
  }
  return fallback;
}

/// Maps an Arrow format string to the DuckDB type the scan will emit.
/// Deliberately narrow: the v1 slice proves the cross-source GPU plan, not a
/// complete Arrow type matrix, and anything outside the allowlist is refused
/// here rather than deep inside the importer.
bool arrow_format_to_duckdb(std::string_view format, duckdb::LogicalType& out)
{
  if (format == "l") {
    out = duckdb::LogicalType::BIGINT;
    return true;
  }
  if (format == "L") {
    out = duckdb::LogicalType::UBIGINT;
    return true;
  }
  if (format == "i") {
    out = duckdb::LogicalType::INTEGER;
    return true;
  }
  if (format == "f") {
    out = duckdb::LogicalType::FLOAT;
    return true;
  }
  if (format == "g") {
    out = duckdb::LogicalType::DOUBLE;
    return true;
  }
  // Both utf8 widths are accepted: cudf imports int64-offset strings natively,
  // and which one a Lance dataset produces is a property of the data.
  if (format == "u" || format == "U") {
    out = duckdb::LogicalType::VARCHAR;
    return true;
  }
  if (format == "tsu:") {
    out = duckdb::LogicalType::TIMESTAMP;
    return true;
  }
  return false;
}

/// Fixed-size list of float32, e.g. "+w:768" — the shape a Lance vector column
/// must have. Returns 0 when the format is not one.
std::size_t fixed_size_list_width(ArrowSchema const& child)
{
  if (child.format == nullptr) { return 0; }
  std::string_view const format{child.format};
  if (!format.starts_with("+w:")) { return 0; }
  if (child.n_children != 1 || child.children == nullptr) { return 0; }
  auto const* values = child.children[0];
  if (values == nullptr || values->format == nullptr) { return 0; }
  std::string_view const value_format{values->format};
  if (value_format != "f" && value_format != "g") { return 0; }
  return static_cast<std::size_t>(std::strtoull(child.format + 3, nullptr, 10));
}

std::vector<float> query_vector_from(duckdb::Value const& value)
{
  std::vector<float> query;
  auto const& children = duckdb::ListValue::GetChildren(value);
  query.reserve(children.size());
  for (auto const& child : children) {
    if (child.IsNull()) {
      throw duckdb::BinderException(
        "sirius_lance_vector_search: the query vector must not contain NULL");
    }
    query.push_back(child.GetValue<float>());
  }
  return query;
}

}  // namespace

duckdb::unique_ptr<duckdb::FunctionData> lance_vector_search_bind(
  duckdb::ClientContext& context,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<std::string>& names)
{
  // Everything that can be judged without touching Lance is judged first, so a
  // rejected query never opens a dataset.
  if (!setting_enabled(context, "sirius_lance_knn_enabled", true)) {
    throw duckdb::BinderException(
      "sirius_lance_vector_search is disabled: set sirius_lance_knn_enabled = true to re-enable "
      "the Lance vector-search scan source");
  }

  if (input.inputs.size() != 3) {
    throw duckdb::BinderException(
      "sirius_lance_vector_search expects (uri, vector_column, query_vector)");
  }
  for (auto const& argument : input.inputs) {
    if (argument.IsNull()) {
      throw duckdb::BinderException("sirius_lance_vector_search: arguments must not be NULL");
    }
  }

  auto bind_data         = duckdb::make_uniq<lance_vector_search_bind_data>();
  auto& spec             = bind_data->spec;
  spec.uri               = input.inputs[0].GetValue<std::string>();
  spec.knn.vector_column = input.inputs[1].GetValue<std::string>();
  spec.knn.query         = query_vector_from(input.inputs[2]);

  std::uint64_t k = 10;
  std::vector<std::string> requested_columns;
  for (auto const& option : input.named_parameters) {
    auto const& key = option.first;
    if (key == "k") {
      auto const raw = option.second.GetValue<std::int64_t>();
      if (raw <= 0) {
        throw duckdb::BinderException("sirius_lance_vector_search: k must be positive, got " +
                                      std::to_string(raw));
      }
      k = static_cast<std::uint64_t>(raw);
    } else if (key == "use_index") {
      spec.knn.use_index = option.second.GetValue<bool>();
    } else if (key == "nprobes") {
      spec.knn.nprobes = static_cast<std::uint64_t>(option.second.GetValue<std::int64_t>());
    } else if (key == "refine_factor") {
      spec.knn.refine_factor = static_cast<std::uint64_t>(option.second.GetValue<std::int64_t>());
    } else if (key == "columns") {
      for (auto const& column : duckdb::ListValue::GetChildren(option.second)) {
        requested_columns.push_back(column.GetValue<std::string>());
      }
    } else if (key == "storage_options") {
      for (auto const& entry : duckdb::MapValue::GetChildren(option.second)) {
        auto const& kv       = duckdb::StructValue::GetChildren(entry);
        auto const opt_key   = kv[0].GetValue<std::string>();
        auto const opt_value = kv[1].GetValue<std::string>();
        auto lowered         = opt_key;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
        if (std::find(kCredentialKeys.begin(), kCredentialKeys.end(), lowered) !=
            kCredentialKeys.end()) {
          throw duckdb::BinderException(
            "sirius_lance_vector_search: storage option '" + opt_key +
            "' carries a credential and must not appear in SQL text; Sirius resolves Lance "
            "credentials from the ambient provider chain instead");
        }
        spec.storage_options.emplace_back(opt_key, opt_value);
      }
    }
  }

  auto const max_k = setting_uint(context, "sirius_lance_max_k", 1000000);
  if (k > max_k) {
    throw duckdb::BinderException("sirius_lance_vector_search: k=" + std::to_string(k) +
                                  " exceeds sirius_lance_max_k=" + std::to_string(max_k));
  }
  spec.knn.k = k;

  // Timeouts are mandatory: an in-flight Lance read cannot be interrupted, so a
  // finite bound is the only thing that keeps a stalled scan from hanging a
  // query indefinitely. A caller may raise them but not remove them.
  auto has_option = [&spec](std::string_view key) {
    return std::any_of(spec.storage_options.begin(),
                       spec.storage_options.end(),
                       [key](auto const& kv) { return kv.first == key; });
  };
  if (!has_option("connect_timeout")) {
    spec.storage_options.emplace_back("connect_timeout", "10s");
  }
  if (!has_option("request_timeout")) {
    spec.storage_options.emplace_back("request_timeout", "30s");
  }

  bind_data->api = acquire_lance_ffi_api();

  lance_error err;
  scoped_dataset dataset(bind_data->api,
                         bind_data->api->dataset_open(spec.uri, spec.storage_options, err));
  if (dataset.get() == nullptr) {
    throw duckdb::BinderException("sirius_lance_vector_search: cannot open Lance dataset '" +
                                  spec.uri + "': " + err.message);
  }

  scoped_arrow_schema schema_owner;
  if (!bind_data->api->knn_schema(dataset.get(), spec.knn, schema_owner.schema, err)) {
    throw duckdb::BinderException("sirius_lance_vector_search: cannot read the KNN schema of '" +
                                  spec.uri + "': " + err.message);
  }
  auto const& schema = schema_owner.schema;

  // The vector column must exist and be a fixed-size list of floats; its width
  // is the contract the query vector has to match.
  std::size_t vector_width  = 0;
  bool vector_column_found  = false;
  std::int64_t vector_index = -1;
  for (std::int64_t i = 0; i < schema.n_children; ++i) {
    auto const& child = *schema.children[i];
    auto const name   = child.name != nullptr ? std::string{child.name} : std::string{};
    if (name != spec.knn.vector_column) { continue; }
    vector_column_found = true;
    vector_index        = i;
    vector_width        = fixed_size_list_width(child);
    if (vector_width == 0) {
      auto const format = child.format != nullptr ? std::string{child.format} : std::string{};
      throw duckdb::BinderException(
        "sirius_lance_vector_search: vector column '" + spec.knn.vector_column +
        "' must be a fixed-size list of floats (Arrow format '+w:<width>'), but its format is '" +
        format + "'");
    }
    break;
  }
  if (!vector_column_found) {
    throw duckdb::BinderException("sirius_lance_vector_search: vector column '" +
                                  spec.knn.vector_column + "' was not found in the dataset schema");
  }
  if (spec.knn.query.size() != vector_width) {
    throw duckdb::BinderException("sirius_lance_vector_search: query vector dimension " +
                                  std::to_string(spec.knn.query.size()) +
                                  " does not match the dimension of '" + spec.knn.vector_column +
                                  "', which is " + std::to_string(vector_width));
  }

  // Every other column is a candidate output. The vector column is skipped by
  // index here, which is exactly what keeps it out of cudf later: the Lance KNN
  // surface has no projection list, so it always arrives in the Arrow batch.
  auto& expect = bind_data->expect;
  for (std::int64_t i = 0; i < schema.n_children; ++i) {
    if (i == vector_index) { continue; }
    auto const& child = *schema.children[i];
    auto const name   = child.name != nullptr ? std::string{child.name} : std::string{};
    auto const format = child.format != nullptr ? std::string{child.format} : std::string{};

    if (!requested_columns.empty() &&
        std::find(requested_columns.begin(), requested_columns.end(), name) ==
          requested_columns.end() &&
        name != "_distance") {
      continue;
    }

    duckdb::LogicalType type;
    if (!arrow_format_to_duckdb(format, type)) {
      throw duckdb::BinderException("sirius_lance_vector_search: column '" + name +
                                    "' has Arrow type '" + format +
                                    "', which this scan source does not support yet");
    }

    expect.kept_child_indices.push_back(static_cast<std::size_t>(i));
    expect.kept_formats.push_back(format);
    expect.kept_names.push_back(name);
    bind_data->names.push_back(name);
    bind_data->return_types.push_back(type);
  }

  if (bind_data->names.empty()) {
    throw duckdb::BinderException(
      "sirius_lance_vector_search: the projection selected no columns from '" + spec.uri + "'");
  }

  names        = duckdb::vector<std::string>(bind_data->names.begin(), bind_data->names.end());
  return_types = duckdb::vector<duckdb::LogicalType>(bind_data->return_types.begin(),
                                                     bind_data->return_types.end());
  return bind_data;
}

void lance_vector_search_execute(duckdb::ClientContext&,
                                 duckdb::TableFunctionInput&,
                                 duckdb::DataChunk&)
{
  // Reached only if the plan was not routed to the GPU scan. This source has no
  // CPU implementation by design, and the no-fallback capability is supposed to
  // surface the original GPU-side error long before this point.
  throw duckdb::NotImplementedException(
    "sirius_lance_vector_search is GPU-only and has no CPU execution path; see the Sirius log "
    "for the GPU-side error");
}

duckdb::unique_ptr<duckdb::NodeStatistics> lance_vector_search_cardinality(
  duckdb::ClientContext&, duckdb::FunctionData const* bind_data)
{
  auto const* data = dynamic_cast<lance_vector_search_bind_data const*>(bind_data);
  if (data == nullptr) { return nullptr; }
  // k is an upper bound on the row count, and a tight one for a populated index.
  return duckdb::make_uniq<duckdb::NodeStatistics>(data->spec.knn.k, data->spec.knn.k);
}

namespace {

/// Plan copies in Sirius are serialize -> deserialize round trips, and
/// LogicalGet::Deserialize re-invokes the bind callback for any table function
/// that ships no serializer. Without these two hooks a single prepare would
/// reopen the dataset once per copy — three times in total — purely to rebuild
/// bind data we already had.
void lance_vector_search_serialize(duckdb::Serializer& serializer,
                                   duckdb::optional_ptr<duckdb::FunctionData> bind_data_p,
                                   duckdb::TableFunction const&)
{
  auto const& bind = bind_data_p->Cast<lance_vector_search_bind_data>();

  // The serializer speaks duckdb::vector and has no pair support, so the
  // storage options travel as two parallel lists.
  duckdb::vector<std::string> option_keys;
  duckdb::vector<std::string> option_values;
  option_keys.reserve(bind.spec.storage_options.size());
  option_values.reserve(bind.spec.storage_options.size());
  for (auto const& [key, value] : bind.spec.storage_options) {
    option_keys.push_back(key);
    option_values.push_back(value);
  }

  duckdb::vector<float> query(bind.spec.knn.query.begin(), bind.spec.knn.query.end());
  duckdb::vector<std::uint64_t> kept_indices(bind.expect.kept_child_indices.begin(),
                                             bind.expect.kept_child_indices.end());
  duckdb::vector<std::string> kept_formats(bind.expect.kept_formats.begin(),
                                           bind.expect.kept_formats.end());
  duckdb::vector<std::string> kept_names(bind.expect.kept_names.begin(),
                                         bind.expect.kept_names.end());
  duckdb::vector<std::string> names(bind.names.begin(), bind.names.end());
  duckdb::vector<duckdb::LogicalType> return_types(bind.return_types.begin(),
                                                   bind.return_types.end());

  serializer.WriteProperty(100, "uri", bind.spec.uri);
  serializer.WriteProperty(101, "option_keys", option_keys);
  serializer.WriteProperty(102, "option_values", option_values);
  serializer.WriteProperty(103, "vector_column", bind.spec.knn.vector_column);
  serializer.WriteProperty(104, "query", query);
  serializer.WriteProperty(105, "k", bind.spec.knn.k);
  serializer.WriteProperty(106, "nprobes", bind.spec.knn.nprobes);
  serializer.WriteProperty(107, "refine_factor", bind.spec.knn.refine_factor);
  serializer.WriteProperty(108, "prefilter", bind.spec.knn.prefilter);
  serializer.WriteProperty(109, "use_index", bind.spec.knn.use_index);
  serializer.WriteProperty(110, "kept_child_indices", kept_indices);
  serializer.WriteProperty(111, "kept_formats", kept_formats);
  serializer.WriteProperty(112, "kept_names", kept_names);
  serializer.WriteProperty(113, "names", names);
  serializer.WriteProperty(114, "return_types", return_types);
}

duckdb::unique_ptr<duckdb::FunctionData> lance_vector_search_deserialize(
  duckdb::Deserializer& deserializer, duckdb::TableFunction&)
{
  auto bind      = duckdb::make_uniq<lance_vector_search_bind_data>();
  bind->spec.uri = deserializer.ReadProperty<std::string>(100, "uri");

  auto const option_keys =
    deserializer.ReadProperty<duckdb::vector<std::string>>(101, "option_keys");
  auto const option_values =
    deserializer.ReadProperty<duckdb::vector<std::string>>(102, "option_values");
  bind->spec.storage_options.reserve(option_keys.size());
  for (std::size_t i = 0; i < option_keys.size() && i < option_values.size(); ++i) {
    bind->spec.storage_options.emplace_back(option_keys[i], option_values[i]);
  }

  bind->spec.knn.vector_column = deserializer.ReadProperty<std::string>(103, "vector_column");
  auto const query             = deserializer.ReadProperty<duckdb::vector<float>>(104, "query");
  bind->spec.knn.query.assign(query.begin(), query.end());
  bind->spec.knn.k             = deserializer.ReadProperty<std::uint64_t>(105, "k");
  bind->spec.knn.nprobes       = deserializer.ReadProperty<std::uint64_t>(106, "nprobes");
  bind->spec.knn.refine_factor = deserializer.ReadProperty<std::uint64_t>(107, "refine_factor");
  bind->spec.knn.prefilter     = deserializer.ReadProperty<bool>(108, "prefilter");
  bind->spec.knn.use_index     = deserializer.ReadProperty<bool>(109, "use_index");

  auto const kept_indices =
    deserializer.ReadProperty<duckdb::vector<std::uint64_t>>(110, "kept_child_indices");
  bind->expect.kept_child_indices.assign(kept_indices.begin(), kept_indices.end());
  auto const kept_formats =
    deserializer.ReadProperty<duckdb::vector<std::string>>(111, "kept_formats");
  bind->expect.kept_formats.assign(kept_formats.begin(), kept_formats.end());
  auto const kept_names = deserializer.ReadProperty<duckdb::vector<std::string>>(112, "kept_names");
  bind->expect.kept_names.assign(kept_names.begin(), kept_names.end());
  auto const names = deserializer.ReadProperty<duckdb::vector<std::string>>(113, "names");
  bind->names.assign(names.begin(), names.end());
  auto const return_types =
    deserializer.ReadProperty<duckdb::vector<duckdb::LogicalType>>(114, "return_types");
  bind->return_types.assign(return_types.begin(), return_types.end());

  // Re-acquired rather than serialized: the handle is process-local, and going
  // through the accessor keeps a test override in effect across plan copies.
  bind->api = acquire_lance_ffi_api();
  return bind;
}

/// Declares two virtual columns, both load-bearing:
///
///  * rowid — so `rowid` resolves in the binder and reaches the planner, which
///    refuses it with a diagnostic explaining that the Lance KNN surface emits
///    no row identity. Without it the binder would just say the column does not
///    exist.
///
///  * the empty column — so that `SELECT count(*)` has a placeholder to carry
///    the row stream. This one is not optional: `LogicalGet::GetAnyColumn`
///    prefers the empty column, then rowid, then column 0, so declaring rowid
///    *alone* makes DuckDB pick rowid as the count(*) placeholder. The scan
///    would then receive a rowid-only projection byte-for-byte identical to the
///    one `SELECT rowid` produces, leaving it unable to tell "count the rows"
///    from "give me the rowids" — and refusing rowid would refuse count(*) with
///    it. Declaring both keeps the two apart. The empty column is invisible to
///    users (duckdb table_binding.cpp skips it during name resolution), and this
///    mirrors what MultiFileReader::GetVirtualColumns does for parquet.
duckdb::virtual_column_map_t lance_vector_search_virtual_columns(
  duckdb::ClientContext&, duckdb::optional_ptr<duckdb::FunctionData>)
{
  duckdb::virtual_column_map_t columns;
  columns.insert(std::make_pair(duckdb::COLUMN_IDENTIFIER_ROW_ID,
                                duckdb::TableColumn("rowid", duckdb::LogicalType::BIGINT)));
  columns.insert(std::make_pair(duckdb::COLUMN_IDENTIFIER_EMPTY,
                                duckdb::TableColumn("", duckdb::LogicalType::BOOLEAN)));
  return columns;
}

}  // namespace

void register_lance_vector_search(duckdb::Catalog& catalog, duckdb::CatalogTransaction transaction)
{
  duckdb::TableFunction function(kLanceVectorSearchName,
                                 {duckdb::LogicalType::VARCHAR,
                                  duckdb::LogicalType::VARCHAR,
                                  duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT)},
                                 lance_vector_search_execute,
                                 lance_vector_search_bind);
  function.named_parameters["k"]             = duckdb::LogicalType::BIGINT;
  function.named_parameters["use_index"]     = duckdb::LogicalType::BOOLEAN;
  function.named_parameters["nprobes"]       = duckdb::LogicalType::BIGINT;
  function.named_parameters["refine_factor"] = duckdb::LogicalType::BIGINT;
  function.named_parameters["columns"] = duckdb::LogicalType::LIST(duckdb::LogicalType::VARCHAR);
  function.named_parameters["storage_options"] =
    duckdb::LogicalType::MAP(duckdb::LogicalType::VARCHAR, duckdb::LogicalType::VARCHAR);
  function.cardinality         = lance_vector_search_cardinality;
  function.get_virtual_columns = lance_vector_search_virtual_columns;
  function.serialize           = lance_vector_search_serialize;
  function.deserialize         = lance_vector_search_deserialize;
  // Landing in the planner's projection-pushdown branch is what lets the scan
  // emit exactly the projected columns.
  function.projection_pushdown = true;
  function.filter_pushdown     = false;

  // Registered as a set rather than a lone function so later ALTER_ON_CONFLICT
  // registrations of the same name add an overload instead of colliding with a
  // single-signature entry.
  duckdb::TableFunctionSet function_set(kLanceVectorSearchName);
  function_set.AddFunction(std::move(function));

  duckdb::CreateTableFunctionInfo info(std::move(function_set));
  info.on_conflict = duckdb::OnCreateConflict::ALTER_ON_CONFLICT;
  catalog.CreateTableFunction(transaction, info);
}

}  // namespace sirius::lance
