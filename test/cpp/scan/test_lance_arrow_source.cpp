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

#include "lance_test_utils.hpp"
#include "test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <op/scan/lance/lance_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace lance = sirius::lance;
namespace scan  = sirius::op::scan;
namespace test  = sirius::test::lance;

constexpr auto kWait = std::chrono::seconds{2};

lance::lance_open_spec open_spec(std::size_t width = 4)
{
  lance::lance_open_spec spec;
  spec.uri               = "memory://fake.lance";
  spec.knn.vector_column = "embedding";
  spec.knn.query.assign(width, 0.25F);
  spec.knn.k = 32;
  return spec;
}

lance::schema_expectation standard_expectation()
{
  return {{0, 1, 3}, {"l", "u", "f"}, {"doc_id", "label", "_distance"}};
}

test::dataset_script batch_script(std::vector<test::stream_event> events,
                                  std::size_t rows         = 4,
                                  std::size_t vector_width = 4)
{
  test::dataset_script script;
  script.schema    = test::make_knn_batch(0, vector_width);
  script.events    = std::move(events);
  script.row_count = static_cast<std::int64_t>(rows);
  return script;
}

std::unique_ptr<scan::lance_ingestible_table_info> ingestible_info(
  std::shared_ptr<test::scripted_lance_ffi> api,
  lance::arrow_batch_retirer_factory retirer_factory = {})
{
  auto info                = std::make_unique<scan::lance_ingestible_table_info>();
  info->api                = std::move(api);
  info->spec               = open_spec();
  info->expect             = standard_expectation();
  info->producer_config    = {.queue_depth = 4, .max_arrow_bytes = 8ULL << 20};
  info->retirer_factory    = std::move(retirer_factory);
  info->names              = {"doc_id", "label", "_distance"};
  info->output_types       = {sirius::logical_type::make(sirius::type_id::BIGINT),
                              sirius::logical_type::make(sirius::type_id::VARCHAR),
                              sirius::logical_type::make(sirius::type_id::FLOAT)};
  info->materialized_order = {0, 1, 2};
  return info;
}

std::unique_ptr<scan::scan_info> run_task(scan::gpu_ingestible::metadata_scan_task_t& task)
{
  if (!task) { return nullptr; }
  return task();
}

void require_error_task(scan::gpu_ingestible::metadata_scan_task_t task, std::string const& message)
{
  REQUIRE(task);
  REQUIRE_THROWS_WITH(task(), Catch::Contains(message));
}

template <typename T>
std::vector<T> download(cudf::column_view const& column, rmm::cuda_stream_view stream)
{
  std::vector<T> values(static_cast<std::size_t>(column.size()));
  if (!values.empty()) {
    cudaMemcpyAsync(values.data(),
                    column.data<T>(),
                    values.size() * sizeof(T),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();
  }
  return values;
}

struct callback_gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered{false};
  bool open{false};
};

void CUDART_CB wait_on_gate(void* opaque)
{
  auto& gate = *static_cast<callback_gate*>(opaque);
  std::unique_lock lock(gate.mutex);
  gate.entered = true;
  gate.cv.notify_all();
  gate.cv.wait(lock, [&] { return gate.open; });
}

bool wait_for_gate(callback_gate& gate)
{
  std::unique_lock lock(gate.mutex);
  return gate.cv.wait_for(lock, kWait, [&] { return gate.entered; });
}

void open_gate(callback_gate& gate)
{
  {
    std::lock_guard lock(gate.mutex);
    gate.open = true;
  }
  gate.cv.notify_all();
}

struct gate_opener {
  callback_gate& gate;
  ~gate_opener() { open_gate(gate); }
};

struct next_call_unblocker {
  std::shared_ptr<test::scripted_lance_ffi> api;
  ~next_call_unblocker() { api->unblock_next_call(); }
};

}  // namespace

TEST_CASE("Lance Arrow batch ownership is move-only and releases exactly once",
          "[lance][arrow][lifecycle]")
{
  auto counters = std::make_shared<test::lifecycle_counters>();
  ArrowArray array{};
  ArrowSchema schema{};
  test::export_batch(test::make_knn_batch(3), array, schema, counters);

  {
    lance::lance_owning_arrow_batch first(std::move(array), std::move(schema));
    REQUIRE(first.valid());
    CHECK(first.num_rows() == 3);

    lance::lance_owning_arrow_batch second(std::move(first));
    CHECK_FALSE(first.valid());
    REQUIRE(second.valid());
    second.release();
    second.release();
    CHECK_FALSE(second.valid());
  }

  CHECK(counters->schema_releases.load() == 1);
  CHECK(counters->array_releases.load() == 1);
}

TEST_CASE("Lance FFI override is scoped and acquired instances outlive the guard",
          "[lance][ffi][lifecycle]")
{
  auto first = std::make_shared<test::scripted_lance_ffi>(
    std::vector{batch_script({test::stream_event::make_eos()})});
  auto second = std::make_shared<test::scripted_lance_ffi>(
    std::vector{batch_script({test::stream_event::make_eos()})});

  std::shared_ptr<lance::lance_ffi_api> retained;
  {
    lance::scoped_ffi_api_override outer([first] { return first; });
    REQUIRE(lance::lance_ffi_api_available());
    retained = lance::acquire_lance_ffi_api();
    CHECK(retained.get() == first.get());

    {
      lance::scoped_ffi_api_override inner([second] { return second; });
      CHECK(lance::acquire_lance_ffi_api().get() == second.get());
    }
    CHECK(lance::acquire_lance_ffi_api().get() == first.get());
  }

  CHECK(retained.get() == first.get());
}

TEST_CASE("Lance producer emits every batch then claims EOF exactly once",
          "[lance][producer][lifecycle]")
{
  auto api = std::make_shared<test::scripted_lance_ffi>(
    std::vector{batch_script({test::stream_event::make_batch(test::make_knn_batch(2, 4, 10)),
                              test::stream_event::make_batch(test::make_knn_batch(3, 4, 20)),
                              test::stream_event::make_eos()},
                             5)});
  auto counters = api->counters();

  {
    lance::lance_split_producer producer(
      api, open_spec(), standard_expectation(), {.queue_depth = 2, .max_arrow_bytes = 1 << 20});

    auto first_task   = producer.claim();
    auto first        = run_task(first_task);
    auto* first_lance = dynamic_cast<scan::lance_scan_info*>(first.get());
    REQUIRE(first_lance);
    CHECK(first_lance->batch.num_rows() == 2);
    CHECK(first_lance->estimated_bytes() > 0);

    auto second_task   = producer.claim();
    auto second        = run_task(second_task);
    auto* second_lance = dynamic_cast<scan::lance_scan_info*>(second.get());
    REQUIRE(second_lance);
    CHECK(second_lance->batch.num_rows() == 3);

    first.reset();
    second.reset();
    CHECK(producer.claim() == nullptr);
    CHECK(producer.eof_claimed());

    auto const stats = producer.snapshot();
    CHECK(stats.batches_produced == 2);
    CHECK(stats.queue_high_water <= 2);
    CHECK(stats.eof_reached);
    CHECK(stats.bytes_produced > 0);
  }

  CHECK(counters->dataset_opens.load() == 1);
  CHECK(counters->dataset_closes.load() == 1);
  CHECK(counters->stream_opens.load() == 1);
  CHECK(counters->stream_closes.load() == 1);
  CHECK(counters->array_releases.load() == 2);
  CHECK(counters->schema_releases.load() == 2);
}

TEST_CASE("Lance producer transports stream-open and stream-next errors once",
          "[lance][producer][error]")
{
  SECTION("stream creation returns null")
  {
    auto script              = batch_script({});
    script.stream_open_error = lance::lance_error{41, "fake stream open failed"};
    auto api                 = std::make_shared<test::scripted_lance_ffi>(
      std::vector<test::dataset_script>{std::move(script)});

    {
      lance::lance_split_producer producer(
        api, open_spec(), standard_expectation(), {.queue_depth = 2, .max_arrow_bytes = 1 << 20});
      require_error_task(producer.claim(), "fake stream open failed");
      CHECK(producer.claim() == nullptr);
    }

    CHECK(api->counters()->dataset_closes.load() == 1);
    CHECK(api->counters()->stream_closes.load() == 0);
  }

  SECTION("stream_next returns an error")
  {
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector{batch_script({test::stream_event::make_error(42, "fake stream next failed")})});

    {
      lance::lance_split_producer producer(
        api, open_spec(), standard_expectation(), {.queue_depth = 2, .max_arrow_bytes = 1 << 20});
      require_error_task(producer.claim(), "fake stream next failed");
      CHECK(producer.claim() == nullptr);
    }

    CHECK(api->counters()->stream_next_calls.load() == 1);
    CHECK(api->counters()->dataset_closes.load() == 1);
    CHECK(api->counters()->stream_closes.load() == 1);
  }
}

TEST_CASE("Lance producer rejects runtime schema drift before handing off a batch",
          "[lance][producer][schema]")
{
  auto drift          = test::make_knn_batch(2);
  drift.mutate_schema = [](ArrowSchema& schema) { schema.children[1]->name = "renamed_label"; };
  auto api            = std::make_shared<test::scripted_lance_ffi>(
    std::vector{batch_script({test::stream_event::make_batch(std::move(drift))})});

  {
    lance::lance_split_producer producer(
      api, open_spec(), standard_expectation(), {.queue_depth = 2, .max_arrow_bytes = 1 << 20});
    require_error_task(producer.claim(), "renamed_label");
    CHECK(producer.claim() == nullptr);
  }

  CHECK(api->counters()->array_releases.load() == 1);
  CHECK(api->counters()->schema_releases.load() == 1);
}

TEST_CASE("Lance producer queue depth bounds run-ahead until a claim",
          "[lance][producer][backpressure]")
{
  std::vector<test::stream_event> events;
  for (std::int64_t i = 0; i < 6; ++i) {
    events.push_back(test::stream_event::make_batch(test::make_knn_batch(1, 4, i)));
  }
  events.push_back(test::stream_event::make_eos());
  auto api =
    std::make_shared<test::scripted_lance_ffi>(std::vector{batch_script(std::move(events), 6)});

  lance::lance_split_producer producer(
    api, open_spec(), standard_expectation(), {.queue_depth = 2, .max_arrow_bytes = 1 << 20});

  REQUIRE(test::wait_until([&] { return api->counters()->stream_next_calls.load() >= 3; }));
  auto const calls_while_full = api->counters()->stream_next_calls.load();
  std::this_thread::sleep_for(std::chrono::milliseconds{30});
  CHECK(api->counters()->stream_next_calls.load() == calls_while_full);
  CHECK(calls_while_full == 3);
  CHECK(producer.snapshot().queue_high_water == 2);

  auto task  = producer.claim();
  auto split = run_task(task);
  REQUIRE(split);
  split.reset();
  REQUIRE(
    test::wait_until([&] { return api->counters()->stream_next_calls.load() > calls_while_full; }));
  producer.stop();
}

TEST_CASE("Lance producer byte cap includes an unprojected vector and fails once",
          "[lance][producer][byte_cap]")
{
  constexpr std::size_t rows  = 64;
  constexpr std::size_t width = 1024;
  constexpr std::uint64_t cap = 4096;
  auto vector_heavy           = test::make_knn_batch(rows, width);
  auto api                    = std::make_shared<test::scripted_lance_ffi>(
    std::vector{batch_script({test::stream_event::make_batch(std::move(vector_heavy)),
                                                 test::stream_event::make_batch(test::make_knn_batch(1, width))},
                             rows,
                             width)});

  lance::schema_expectation expectation{{0, 3}, {"l", "f"}, {"doc_id", "_distance"}};
  {
    lance::lance_split_producer producer(
      api, open_spec(width), std::move(expectation), {.queue_depth = 4, .max_arrow_bytes = cap});
    require_error_task(producer.claim(), "Arrow byte cap");
    CHECK(producer.claim() == nullptr);

    auto const stats = producer.snapshot();
    CHECK(stats.byte_cap_tripped);
    CHECK(stats.bytes_produced > cap);
    CHECK(stats.batches_produced == 0);
  }

  CHECK(api->counters()->stream_next_calls.load() == 1);
  CHECK(api->counters()->array_releases.load() == 1);
  CHECK(api->counters()->schema_releases.load() == 1);
}

TEST_CASE("Lance producer stop waits for one in-flight FFI call then joins cleanly",
          "[lance][producer][cancellation]")
{
  auto api = std::make_shared<test::scripted_lance_ffi>(
    std::vector{batch_script({test::stream_event::make_batch(test::make_knn_batch(1))})});
  api->block_next_call();

  {
    lance::lance_split_producer producer(
      api, open_spec(), standard_expectation(), {.queue_depth = 2, .max_arrow_bytes = 1 << 20});
    next_call_unblocker ensure_unblocked{api};
    REQUIRE(api->wait_until_next_blocked(kWait));

    auto stopped = std::async(std::launch::async, [&] {
      producer.stop();
      return true;
    });
    CHECK(stopped.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout);
    api->unblock_next_call();
    REQUIRE(stopped.wait_for(kWait) == std::future_status::ready);
    CHECK(stopped.get());
    producer.stop();
  }

  CHECK(api->counters()->dataset_closes.load() == 1);
  CHECK(api->counters()->stream_closes.load() == 1);
  CHECK(api->counters()->array_releases.load() == api->counters()->schema_releases.load());
}

TEST_CASE("Lance ingestible imports scalar Arrow children and never imports the vector",
          "[lance][ingestible][gpu]")
{
  SECTION("nullable scalar output")
  {
    auto api      = std::make_shared<test::scripted_lance_ffi>(std::vector{batch_script(
      {test::stream_event::make_batch(test::make_knn_batch(3)), test::stream_event::make_eos()},
      3)});
    auto counters = api->counters();
    scan::lance_gpu_ingestible ingestible(
      ingestible_info(api, [] { return lance::make_synchronizing_retirer(); }));

    auto const names = ingestible.table_info().column_names();
    REQUIRE(names.size() == 3);
    CHECK(names[0] == "doc_id");
    CHECK(names[1] == "label");
    CHECK(names[2] == "_distance");
    CHECK(ingestible.table_info().file_paths().size() == 1);
    CHECK(ingestible.materialized_column_order() == std::vector<std::size_t>{0, 1, 2});

    auto task = ingestible.next_split_provider(
      [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return nullptr; });
    REQUIRE(task);
    auto split = task();
    REQUIRE(dynamic_cast<scan::lance_scan_info*>(split.get()));

    auto manager    = initialize_memory_manager();
    auto* gpu_space = sirius::scan_test_utils::get_space(*manager, cucascade::memory::Tier::GPU);
    REQUIRE(gpu_space);
    rmm::cuda_stream stream;

    scan::scan_operator_input input(std::move(split));
    input.gpu_memory_space = gpu_space;
    auto result            = ingestible.materialize_table(input, stream.view());
    auto view              = result.table.view();
    stream.synchronize();

    REQUIRE(view.num_columns() == 3);
    REQUIRE(view.num_rows() == 3);
    CHECK(view.column(0).type().id() == cudf::type_id::INT64);
    CHECK(view.column(1).type().id() == cudf::type_id::STRING);
    CHECK(view.column(2).type().id() == cudf::type_id::FLOAT32);
    CHECK(view.column(1).null_count() == 1);
    CHECK(download<std::int64_t>(view.column(0), stream.view()) ==
          std::vector<std::int64_t>{0, 1, 2});
    CHECK(download<float>(view.column(2), stream.view()) == std::vector<float>{0.0F, 0.1F, 0.2F});
    CHECK(counters->array_releases.load() == 1);
    CHECK(counters->schema_releases.load() == 1);

    auto projected =
      ingestible.post_filter_and_project(std::move(result), *gpu_space, stream.view());
    REQUIRE(projected);
    CHECK(projected->num_columns() == 3);
    CHECK(projected->num_rows() == 3);
  }

  SECTION("empty stream emits one typed zero-row sentinel")
  {
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector{batch_script({test::stream_event::make_eos()}, 0)});
    auto ingestible = scan::make_ingestible(
      ingestible_info(api, [] { return lance::make_synchronizing_retirer(); }));

    auto task = ingestible->next_split_provider(
      [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return nullptr; });
    CHECK_FALSE(task);
    CHECK(ingestible->has_processed_all_metadata());
    auto coalescer = ingestible->create_batch_coalescer();
    auto tail      = coalescer->flush();
    REQUIRE(tail.size() == 1);
    auto split        = std::move(tail.front());
    auto* lance_split = dynamic_cast<scan::lance_scan_info*>(split.get());
    REQUIRE(lance_split);
    REQUIRE(lance_split->is_empty_sentinel);

    auto manager    = initialize_memory_manager();
    auto* gpu_space = sirius::scan_test_utils::get_space(*manager, cucascade::memory::Tier::GPU);
    REQUIRE(gpu_space);
    rmm::cuda_stream stream;
    scan::scan_operator_input input(std::move(split));
    input.gpu_memory_space = gpu_space;
    auto result            = ingestible->materialize_table(input, stream.view());
    CHECK(result.table.view().num_columns() == 3);
    CHECK(result.table.view().num_rows() == 0);
  }
}

TEST_CASE("Lance Arrow retirement synchronizes success and conversion-failure paths",
          "[lance][ingestible][retirement][gpu]")
{
  SECTION("default retirer does not release before its stream is ready")
  {
    auto counters = std::make_shared<test::lifecycle_counters>();
    ArrowArray array{};
    ArrowSchema schema{};
    test::export_batch(test::make_knn_batch(1), array, schema, counters);
    lance::lance_owning_arrow_batch batch(std::move(array), std::move(schema));
    auto retirer = lance::make_synchronizing_retirer();
    rmm::cuda_stream stream;
    callback_gate gate;
    REQUIRE(cudaLaunchHostFunc(stream.value(), wait_on_gate, &gate) == cudaSuccess);

    auto retired = std::async(std::launch::async, [&] {
      retirer->retire(std::move(batch), stream.view());
      return true;
    });
    gate_opener ensure_open{gate};
    REQUIRE(wait_for_gate(gate));
    CHECK(retired.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout);
    CHECK(counters->array_releases.load() == 0);
    CHECK(counters->schema_releases.load() == 0);

    open_gate(gate);
    REQUIRE(retired.wait_for(kWait) == std::future_status::ready);
    CHECK(retired.get());
    CHECK(counters->array_releases.load() == 1);
    CHECK(counters->schema_releases.load() == 1);
  }

  SECTION("GPU OOM synchronizes before the scan-info host owner is released")
  {
    auto api = std::make_shared<test::scripted_lance_ffi>(
      std::vector{batch_script({test::stream_event::make_eos()})});
    scan::lance_gpu_ingestible ingestible(
      ingestible_info(api, [] { return lance::make_synchronizing_retirer(); }));
    auto counters = api->counters();

    ArrowArray array{};
    ArrowSchema schema{};
    test::export_batch(test::make_knn_batch(2048), array, schema, counters);
    auto split           = std::make_unique<scan::lance_scan_info>();
    split->batch         = lance::lance_owning_arrow_batch(std::move(array), std::move(schema));
    split->decoded_bytes = 2048 * (sizeof(std::int64_t) + sizeof(float));

    cucascade::memory::gpu_memory_space_config config;
    config.device_id                  = 0;
    config.memory_capacity            = 4096;
    config.reservation_limit_fraction = 1.0;
    config.per_stream_reservation     = false;
    cucascade::memory::memory_space tiny_gpu(config);
    rmm::cuda_stream stream;
    callback_gate gate;
    REQUIRE(cudaLaunchHostFunc(stream.value(), wait_on_gate, &gate) == cudaSuccess);

    auto failed = std::async(std::launch::async, [&] {
      try {
        auto ignored = ingestible.materialize_metadata_to_table(*split, tiny_gpu, stream.view());
        return false;
      } catch (rmm::bad_alloc const&) {
        return true;
      }
    });
    gate_opener ensure_open{gate};

    REQUIRE(wait_for_gate(gate));
    CHECK(failed.wait_for(std::chrono::milliseconds{50}) == std::future_status::timeout);
    CHECK(counters->array_releases.load() == 0);
    open_gate(gate);
    REQUIRE(failed.wait_for(kWait) == std::future_status::ready);
    REQUIRE(failed.get());

    split.reset();
    CHECK(counters->array_releases.load() == 1);
    CHECK(counters->schema_releases.load() == 1);
  }
}
