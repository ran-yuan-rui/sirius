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

// This translation unit is the only place that touches the pinned Rust
// staticlib. It is compiled solely when SIRIUS_ENABLE_LANCE_KNN=ON, which also
// defines SIRIUS_HAVE_LANCE_KNN; everything else in the Lance scan source works
// against the abstract lance_ffi_api and builds unconditionally.
//
// Must precede any other Arrow-adjacent header: lance_ffi_api.hpp pulls in
// DuckDB's arrow.hpp, whose whole-file guard is ARROW_FLAG_DICTIONARY_ORDERED.
#include <lance_shim/lance_ffi_api.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::lance {

namespace {

//===----------------------------------------------------------------------===//
// Pinned C surface
//===----------------------------------------------------------------------===//
/// Mirrors lance-duckdb `main@6316cbb`, `src/include/lance_ffi.hpp`. Declared
/// here rather than included so the Rust checkout is not a compile-time include
/// dependency of Sirius; the pin is enforced by cmake/lance_ffi.cmake, and any
/// signature drift becomes a link error rather than silent UB.
extern "C" {
void* lance_open_dataset_with_storage_options(char const* path,
                                              char const** option_keys,
                                              char const** option_values,
                                              std::size_t options_len);
void lance_close_dataset(void* dataset);
std::int64_t lance_dataset_count_rows(void* dataset);

void* lance_get_knn_schema(void* dataset,
                           char const* vector_column,
                           float const* query_values,
                           std::size_t query_len,
                           std::uint64_t k,
                           std::uint64_t nprobes,
                           std::uint64_t refine_factor,
                           std::uint8_t prefilter,
                           std::uint8_t use_index);
void lance_free_schema(void* schema);
std::int32_t lance_schema_to_arrow(void* schema, ArrowSchema* out_schema);

void* lance_create_knn_stream_ir(void* dataset,
                                 char const* vector_column,
                                 float const* query_values,
                                 std::size_t query_len,
                                 std::uint64_t k,
                                 std::uint64_t nprobes,
                                 std::uint64_t refine_factor,
                                 std::uint8_t const* filter_ir,
                                 std::size_t filter_ir_len,
                                 std::uint8_t prefilter,
                                 std::uint8_t use_index);
std::int32_t lance_stream_next(void* stream, void** out_batch);
void lance_close_stream(void* stream);
std::int32_t lance_batch_to_arrow(void* batch, ArrowArray* out_array, ArrowSchema* out_schema);
void lance_free_batch(void* batch);

std::int32_t lance_last_error_code();
char const* lance_last_error_message();
void lance_free_string(char const* s);
}  // extern "C"

//===----------------------------------------------------------------------===//
// error capture
//===----------------------------------------------------------------------===//
/**
 * @brief Drains the Rust error slot into a caller-owned struct.
 *
 * The slot is a `thread_local` in the Rust crate and reading the message hands
 * over an owned string, so this must run on the very thread whose call failed,
 * immediately after it, and must free what it read. Every call site below obeys
 * that by capturing inside the same lambda that made the call.
 */
void capture_error(lance_error& err)
{
  err.code        = lance_last_error_code();
  char const* msg = lance_last_error_message();
  if (msg != nullptr) {
    err.message.assign(msg);
    lance_free_string(msg);
  } else {
    err.message.clear();
  }
  // A failed call with an empty slot would otherwise look like success to
  // lance_error::ok(); give the caller something actionable instead.
  if (err.code == 0 && err.message.empty()) {
    err.code    = -1;
    err.message = "Lance reported a failure but left no diagnostic";
  }
}

//===----------------------------------------------------------------------===//
// control thread
//===----------------------------------------------------------------------===//
/**
 * @brief A single resident thread that owns every dataset-class Lance call.
 *
 * Why this exists: the crate is built with `panic="abort"` and its blocking
 * entry points run `runtime::block_on`, so calling one from a thread that is
 * already a Tokio worker aborts the process. Sirius cannot rule that out — a
 * host embedding DuckDB inside its own Tokio runtime would do exactly that on
 * the bind path. Routing through a thread this shim created makes the caller's
 * identity irrelevant.
 *
 * Stream-class calls deliberately do NOT come through here: they are only ever
 * issued from lance_split_producer's own std::thread, which is likewise never a
 * Tokio worker, and the stream must stay pinned to one thread. Keeping them
 * inline also avoids a thread handoff per record batch.
 */
class control_thread {
 public:
  control_thread() : _worker([this] { run(); }) {}

  ~control_thread()
  {
    {
      std::lock_guard lock(_mutex);
      _stopping = true;
    }
    _wake.notify_all();
    if (_worker.joinable()) { _worker.join(); }
  }

  control_thread(control_thread const&)            = delete;
  control_thread& operator=(control_thread const&) = delete;

  /// Runs @p work on the control thread and blocks until it returns. Exceptions
  /// escaping @p work are rethrown on the caller.
  template <typename Fn>
  auto run_sync(Fn&& work) -> decltype(work())
  {
    using result_t = decltype(work());

    struct slot {
      std::mutex mutex;
      std::condition_variable done;
      bool finished{false};
      std::exception_ptr failure;
      std::optional<result_t> value;
    };
    auto cell = std::make_shared<slot>();

    submit([cell, work = std::forward<Fn>(work)]() mutable {
      try {
        cell->value.emplace(work());
      } catch (...) {
        cell->failure = std::current_exception();
      }
      {
        std::lock_guard lock(cell->mutex);
        cell->finished = true;
      }
      cell->done.notify_one();
    });

    std::unique_lock lock(cell->mutex);
    cell->done.wait(lock, [&] { return cell->finished; });
    if (cell->failure) { std::rethrow_exception(cell->failure); }
    return std::move(*cell->value);
  }

 private:
  void submit(std::function<void()> work)
  {
    {
      std::lock_guard lock(_mutex);
      if (_stopping) {
        throw std::runtime_error("Lance FFI control thread is shutting down");
      }
      _queue.push_back(std::move(work));
    }
    _wake.notify_one();
  }

  void run()
  {
    for (;;) {
      std::function<void()> work;
      {
        std::unique_lock lock(_mutex);
        _wake.wait(lock, [&] { return _stopping || !_queue.empty(); });
        // Drain before honouring the stop: a queued caller is already blocked
        // waiting on its slot and would never be woken otherwise.
        if (_queue.empty()) { return; }
        work = std::move(_queue.front());
        _queue.pop_front();
      }
      work();
    }
  }

  std::mutex _mutex;
  std::condition_variable _wake;
  std::deque<std::function<void()>> _queue;
  bool _stopping{false};
  std::thread _worker;
};

//===----------------------------------------------------------------------===//
// embedded_lance_ffi
//===----------------------------------------------------------------------===//
class embedded_lance_ffi final : public lance_ffi_api {
 public:
  void* dataset_open(std::string_view uri,
                     storage_option_list const& storage_options,
                     lance_error& err) override
  {
    // The C surface takes NUL-terminated strings and two parallel arrays whose
    // elements must outlive the call, so everything is materialized up front.
    std::string const path{uri};
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.reserve(storage_options.size());
    values.reserve(storage_options.size());
    for (auto const& [key, value] : storage_options) {
      keys.push_back(key);
      values.push_back(value);
    }

    auto outcome = _control.run_sync([&]() -> std::pair<void*, lance_error> {
      std::vector<char const*> key_ptrs;
      std::vector<char const*> value_ptrs;
      key_ptrs.reserve(keys.size());
      value_ptrs.reserve(values.size());
      for (std::size_t i = 0; i < keys.size(); ++i) {
        key_ptrs.push_back(keys[i].c_str());
        value_ptrs.push_back(values[i].c_str());
      }

      lance_error local;
      auto* handle = lance_open_dataset_with_storage_options(
        path.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(),
        key_ptrs.size());
      if (handle == nullptr) { capture_error(local); }
      return {handle, std::move(local)};
    });

    if (outcome.first == nullptr) { err = std::move(outcome.second); }
    return outcome.first;
  }

  void dataset_close(void* dataset) noexcept override
  {
    if (dataset == nullptr) { return; }
    try {
      _control.run_sync([dataset]() -> int {
        lance_close_dataset(dataset);
        return 0;
      });
    } catch (...) {
      // Nothing useful remains: this runs on teardown paths that must not throw.
      // The handle leaks rather than risking a call on an unknown thread.
    }
  }

  std::int64_t dataset_count_rows(void* dataset, lance_error& err) override
  {
    auto outcome = _control.run_sync([dataset]() -> std::pair<std::int64_t, lance_error> {
      lance_error local;
      auto const rows = lance_dataset_count_rows(dataset);
      if (rows < 0) { capture_error(local); }
      return {rows, std::move(local)};
    });
    if (outcome.first < 0) { err = std::move(outcome.second); }
    return outcome.first;
  }

  bool knn_schema(void* dataset, knn_params const& p, ArrowSchema& out, lance_error& err) override
  {
    auto outcome = _control.run_sync([&]() -> std::pair<bool, lance_error> {
      lance_error local;
      void* schema = lance_get_knn_schema(dataset,
                                          p.vector_column.c_str(),
                                          p.query.data(),
                                          p.query.size(),
                                          p.k,
                                          p.nprobes,
                                          p.refine_factor,
                                          static_cast<std::uint8_t>(p.prefilter),
                                          static_cast<std::uint8_t>(p.use_index));
      if (schema == nullptr) {
        capture_error(local);
        return {false, std::move(local)};
      }
      // The handle is an intermediate: export it into the caller's struct and
      // free it here so no Lance-owned schema handle ever escapes this TU.
      auto const rc = lance_schema_to_arrow(schema, &out);
      lance_free_schema(schema);
      if (rc != 0) {
        capture_error(local);
        return {false, std::move(local)};
      }
      return {true, std::move(local)};
    });

    if (!outcome.first) { err = std::move(outcome.second); }
    return outcome.first;
  }

  // Stream-class calls run inline; see control_thread's comment for why that is
  // both safe and necessary.
  void* knn_stream_open(void* dataset, knn_params const& p, lance_error& err) override
  {
    // v1 registers the table function without filter pushdown, so there is no
    // filter IR to hand down.
    auto* stream = lance_create_knn_stream_ir(dataset,
                                              p.vector_column.c_str(),
                                              p.query.data(),
                                              p.query.size(),
                                              p.k,
                                              p.nprobes,
                                              p.refine_factor,
                                              nullptr,
                                              0,
                                              static_cast<std::uint8_t>(p.prefilter),
                                              static_cast<std::uint8_t>(p.use_index));
    if (stream == nullptr) { capture_error(err); }
    return stream;
  }

  std::int32_t stream_next(void* stream,
                           ArrowArray& out_array,
                           ArrowSchema& out_schema,
                           lance_error& err) override
  {
    void* batch   = nullptr;
    auto const rc = lance_stream_next(stream, &batch);
    if (rc == 1) { return 1; }
    if (rc != 0) {
      capture_error(err);
      return -1;
    }
    if (batch == nullptr) {
      err.code    = -1;
      err.message = "Lance signalled a batch but produced none";
      return -1;
    }

    // Export then free unconditionally: the batch handle must not outlive this
    // call even when the export fails, or it leaks on the error path.
    auto const exported = lance_batch_to_arrow(batch, &out_array, &out_schema);
    lance_free_batch(batch);
    if (exported != 0) {
      capture_error(err);
      return -1;
    }
    return 0;
  }

  void stream_close(void* stream) noexcept override
  {
    if (stream != nullptr) { lance_close_stream(stream); }
  }

 private:
  control_thread _control;
};

}  // namespace

std::shared_ptr<lance_ffi_api> make_embedded_lance_ffi_api()
{
  // One control thread per API instance, created with it. acquire_lance_ffi_api
  // calls the factory outside its lock precisely so this construction may block.
  return std::make_shared<embedded_lance_ffi>();
}

}  // namespace sirius::lance
