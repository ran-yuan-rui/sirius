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

#pragma once

// This must be the first Arrow-adjacent include. See lance_ffi_api.hpp.
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/arrow/arrow_converter.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/database.hpp>
#include <lance_shim/lance_ffi_api.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test::lance {

namespace detail {

class scoped_sirius_disable {
 public:
  scoped_sirius_disable()
  {
    if (auto const* current = std::getenv("SIRIUS_DISABLE")) { _previous = current; }
    setenv("SIRIUS_DISABLE", "1", 1);
  }

  ~scoped_sirius_disable()
  {
    if (_previous) {
      setenv("SIRIUS_DISABLE", _previous->c_str(), 1);
    } else {
      unsetenv("SIRIUS_DISABLE");
    }
  }

 private:
  std::optional<std::string> _previous;
};

struct arrow_export_context {
  arrow_export_context()
  {
    scoped_sirius_disable disable;
    db         = std::make_unique<duckdb::DuckDB>(nullptr);
    connection = std::make_unique<duckdb::Connection>(*db);
    properties = connection->context->GetClientProperties();
  }

  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> connection;
  duckdb::ClientProperties properties;
};

}  // namespace detail

/// ArrowConverter::ToArrowSchema dereferences options.client_context, so a
/// default-constructed ClientProperties object is not a valid export context.
inline duckdb::ClientProperties const& arrow_export_properties()
{
  static detail::arrow_export_context context;
  return context.properties;
}

struct lifecycle_counters {
  std::atomic<std::uint64_t> array_releases{0};
  std::atomic<std::uint64_t> schema_releases{0};
  std::atomic<std::uint64_t> dataset_opens{0};
  std::atomic<std::uint64_t> dataset_closes{0};
  std::atomic<std::uint64_t> stream_opens{0};
  std::atomic<std::uint64_t> stream_closes{0};
  std::atomic<std::uint64_t> stream_next_calls{0};
  std::atomic<std::uint64_t> cpu_callbacks{0};
};

struct arrow_batch_spec {
  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<std::string> names;
  std::vector<std::vector<duckdb::Value>> rows;
  std::function<void(ArrowSchema&)> mutate_schema;
};

inline duckdb::Value float_array(std::size_t width, float base = 0.0F)
{
  duckdb::vector<duckdb::Value> values;
  values.reserve(width);
  for (std::size_t i = 0; i < width; ++i) {
    values.push_back(duckdb::Value::FLOAT(base + static_cast<float>(i) / 1000.0F));
  }
  return duckdb::Value::ARRAY(duckdb::LogicalType::FLOAT, std::move(values));
}

inline arrow_batch_spec make_knn_batch(std::size_t rows,
                                       std::size_t vector_width = 4,
                                       std::int64_t id_base     = 0)
{
  arrow_batch_spec spec;
  spec.types = {duckdb::LogicalType::BIGINT,
                duckdb::LogicalType::VARCHAR,
                duckdb::LogicalType::ARRAY(duckdb::LogicalType::FLOAT, vector_width),
                duckdb::LogicalType::FLOAT};
  spec.names = {"doc_id", "label", "embedding", "_distance"};
  spec.rows.reserve(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    auto const id = id_base + static_cast<std::int64_t>(i);
    spec.rows.push_back({duckdb::Value::BIGINT(id),
                         i % 3 == 1 ? duckdb::Value(duckdb::LogicalType::VARCHAR)
                                    : duckdb::Value("label_" + std::to_string(id)),
                         float_array(vector_width, static_cast<float>(i)),
                         duckdb::Value::FLOAT(static_cast<float>(i) / 10.0F)});
  }
  return spec;
}

namespace detail {

struct array_release_state {
  ArrowArray inner{};
  std::shared_ptr<lifecycle_counters> counters;
};

struct schema_release_state {
  ArrowSchema inner{};
  std::shared_ptr<lifecycle_counters> counters;
};

inline void counted_array_release(ArrowArray* array)
{
  if (!array || !array->release) { return; }
  auto* state         = static_cast<array_release_state*>(array->private_data);
  array->release      = nullptr;
  array->private_data = nullptr;
  if (state->inner.release) { state->inner.release(&state->inner); }
  state->counters->array_releases.fetch_add(1, std::memory_order_relaxed);
  delete state;
}

inline void counted_schema_release(ArrowSchema* schema)
{
  if (!schema || !schema->release) { return; }
  auto* state          = static_cast<schema_release_state*>(schema->private_data);
  schema->release      = nullptr;
  schema->private_data = nullptr;
  if (state->inner.release) { state->inner.release(&state->inner); }
  state->counters->schema_releases.fetch_add(1, std::memory_order_relaxed);
  delete state;
}

inline void count_releases(ArrowArray& array,
                           ArrowSchema& schema,
                           std::shared_ptr<lifecycle_counters> counters)
{
  auto* array_state   = new array_release_state{array, counters};
  array.release       = counted_array_release;
  array.private_data  = array_state;
  auto* schema_state  = new schema_release_state{schema, std::move(counters)};
  schema.release      = counted_schema_release;
  schema.private_data = schema_state;
}

inline void count_release(ArrowSchema& schema, std::shared_ptr<lifecycle_counters> counters)
{
  auto* state         = new schema_release_state{schema, std::move(counters)};
  schema.release      = counted_schema_release;
  schema.private_data = state;
}

inline std::unique_ptr<duckdb::DataChunk> make_chunk(arrow_batch_spec const& spec)
{
  auto chunk = std::make_unique<duckdb::DataChunk>();
  chunk->Initialize(duckdb::Allocator::DefaultAllocator(), spec.types);
  chunk->SetCardinality(spec.rows.size());
  for (duckdb::idx_t row = 0; row < spec.rows.size(); ++row) {
    if (spec.rows[row].size() != spec.types.size()) {
      throw std::invalid_argument("fake Lance row width does not match schema");
    }
    for (duckdb::idx_t col = 0; col < spec.types.size(); ++col) {
      chunk->SetValue(col, row, spec.rows[row][col]);
    }
  }
  return chunk;
}

}  // namespace detail

inline void export_schema(arrow_batch_spec const& spec,
                          ArrowSchema& out,
                          std::shared_ptr<lifecycle_counters> counters)
{
  auto properties = arrow_export_properties();
  duckdb::ArrowConverter::ToArrowSchema(&out, spec.types, spec.names, properties);
  if (spec.mutate_schema) { spec.mutate_schema(out); }
  detail::count_release(out, std::move(counters));
}

inline void export_batch(arrow_batch_spec const& spec,
                         ArrowArray& out_array,
                         ArrowSchema& out_schema,
                         std::shared_ptr<lifecycle_counters> counters)
{
  auto chunk      = detail::make_chunk(spec);
  auto properties = arrow_export_properties();
  duckdb::unordered_map<duckdb::idx_t, const duckdb::shared_ptr<duckdb::ArrowTypeExtensionData>>
    extension_types;
  duckdb::ArrowConverter::ToArrowArray(*chunk, &out_array, properties, extension_types);
  duckdb::ArrowConverter::ToArrowSchema(&out_schema, spec.types, spec.names, properties);
  if (spec.mutate_schema) { spec.mutate_schema(out_schema); }
  detail::count_releases(out_array, out_schema, std::move(counters));
}

inline void release_exported(ArrowArray& array, ArrowSchema& schema)
{
  if (schema.release) { schema.release(&schema); }
  if (array.release) { array.release(&array); }
}

enum class event_kind { BATCH, ERROR, EOS };

struct stream_event {
  event_kind kind{event_kind::EOS};
  arrow_batch_spec batch;
  sirius::lance::lance_error error;

  static stream_event make_batch(arrow_batch_spec value)
  {
    stream_event event;
    event.kind  = event_kind::BATCH;
    event.batch = std::move(value);
    return event;
  }

  static stream_event make_error(std::int32_t code, std::string message)
  {
    stream_event event;
    event.kind  = event_kind::ERROR;
    event.error = {code, std::move(message)};
    return event;
  }

  static stream_event make_eos() { return {}; }
};

struct dataset_script {
  arrow_batch_spec schema;
  std::vector<stream_event> events;
  std::int64_t row_count{0};
  std::optional<sirius::lance::lance_error> open_error;
  std::optional<sirius::lance::lance_error> count_error;
  std::optional<sirius::lance::lance_error> schema_error;
  std::optional<sirius::lance::lance_error> stream_open_error;
};

class scripted_lance_ffi final : public sirius::lance::lance_ffi_api {
 public:
  explicit scripted_lance_ffi(
    std::vector<dataset_script> scripts,
    std::shared_ptr<lifecycle_counters> counters = std::make_shared<lifecycle_counters>())
    : _counters(std::move(counters))
  {
    if (scripts.empty()) { throw std::invalid_argument("fake Lance API needs one script"); }
    _scripts.reserve(scripts.size());
    for (auto& script : scripts) {
      _scripts.push_back(std::make_shared<dataset_script const>(std::move(script)));
    }
  }

  std::shared_ptr<lifecycle_counters> counters() const { return _counters; }

  void block_next_call()
  {
    std::lock_guard lock(_block_mutex);
    _block_enabled = true;
    _block_entered = false;
    _block_open    = false;
  }

  bool wait_until_next_blocked(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock(_block_mutex);
    return _block_cv.wait_for(lock, timeout, [&] { return _block_entered; });
  }

  void unblock_next_call()
  {
    {
      std::lock_guard lock(_block_mutex);
      _block_open = true;
    }
    _block_cv.notify_all();
  }

  void* dataset_open(std::string_view,
                     sirius::lance::storage_option_list const&,
                     sirius::lance::lance_error& err) override
  {
    auto const ordinal = _counters->dataset_opens.fetch_add(1, std::memory_order_relaxed);
    auto script        = script_for(ordinal);
    if (script->open_error) {
      err = *script->open_error;
      return nullptr;
    }
    err.clear();
    return new dataset_handle{std::move(script), ordinal};
  }

  void dataset_close(void* dataset) noexcept override
  {
    if (!dataset) { return; }
    delete static_cast<dataset_handle*>(dataset);
    _counters->dataset_closes.fetch_add(1, std::memory_order_relaxed);
  }

  std::int64_t dataset_count_rows(void* dataset, sirius::lance::lance_error& err) override
  {
    auto const& script = checked_dataset(dataset).script;
    if (script->count_error) {
      err = *script->count_error;
      return -1;
    }
    err.clear();
    return script->row_count;
  }

  bool knn_schema(void* dataset,
                  sirius::lance::knn_params const&,
                  ArrowSchema& out,
                  sirius::lance::lance_error& err) override
  {
    auto const& script = checked_dataset(dataset).script;
    if (script->schema_error) {
      err = *script->schema_error;
      return false;
    }
    export_schema(script->schema, out, _counters);
    err.clear();
    return true;
  }

  void* knn_stream_open(void* dataset,
                        sirius::lance::knn_params const&,
                        sirius::lance::lance_error& err) override
  {
    auto const& handle = checked_dataset(dataset);
    if (handle.script->stream_open_error) {
      err = *handle.script->stream_open_error;
      return nullptr;
    }
    _counters->stream_opens.fetch_add(1, std::memory_order_relaxed);
    err.clear();
    return new stream_handle{handle.script, 0};
  }

  std::int32_t stream_next(void* stream,
                           ArrowArray& out_array,
                           ArrowSchema& out_schema,
                           sirius::lance::lance_error& err) override
  {
    _counters->stream_next_calls.fetch_add(1, std::memory_order_relaxed);
    maybe_block();

    auto& handle = checked_stream(stream);
    if (handle.next_event >= handle.script->events.size()) {
      err.clear();
      return 1;
    }

    auto const& event = handle.script->events[handle.next_event++];
    switch (event.kind) {
      case event_kind::BATCH:
        export_batch(event.batch, out_array, out_schema, _counters);
        err.clear();
        return 0;
      case event_kind::ERROR: err = event.error; return -1;
      case event_kind::EOS: err.clear(); return 1;
    }
    throw std::logic_error("unreachable fake Lance event");
  }

  void stream_close(void* stream) noexcept override
  {
    if (!stream) { return; }
    delete static_cast<stream_handle*>(stream);
    _counters->stream_closes.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  struct dataset_handle {
    std::shared_ptr<dataset_script const> script;
    std::uint64_t ordinal;
  };

  struct stream_handle {
    std::shared_ptr<dataset_script const> script;
    std::size_t next_event;
  };

  std::shared_ptr<dataset_script const> script_for(std::uint64_t ordinal) const
  {
    auto const index = std::min<std::size_t>(ordinal, _scripts.size() - 1);
    return _scripts[index];
  }

  static dataset_handle& checked_dataset(void* dataset)
  {
    if (!dataset) { throw std::invalid_argument("null fake Lance dataset"); }
    return *static_cast<dataset_handle*>(dataset);
  }

  static stream_handle& checked_stream(void* stream)
  {
    if (!stream) { throw std::invalid_argument("null fake Lance stream"); }
    return *static_cast<stream_handle*>(stream);
  }

  void maybe_block()
  {
    std::unique_lock lock(_block_mutex);
    if (!_block_enabled) { return; }
    _block_enabled = false;
    _block_entered = true;
    _block_cv.notify_all();
    _block_cv.wait(lock, [&] { return _block_open; });
  }

  std::vector<std::shared_ptr<dataset_script const>> _scripts;
  std::shared_ptr<lifecycle_counters> _counters;
  std::mutex _block_mutex;
  std::condition_variable _block_cv;
  bool _block_enabled{false};
  bool _block_entered{false};
  bool _block_open{false};
};

template <typename Predicate>
bool wait_until(Predicate&& predicate,
                std::chrono::milliseconds timeout = std::chrono::milliseconds{2000})
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  }
  return true;
}

}  // namespace sirius::test::lance
