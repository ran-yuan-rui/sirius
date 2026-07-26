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

#include <op/scan/lance/lance_gpu_ingestible.hpp>
#include <op/scan/lance/lance_split_producer.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace sirius::lance {

namespace {

/// Adds without wrapping: the byte cap must stay meaningful even if a
/// pathological batch reports an absurd size.
std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) noexcept
{
  auto const limit = std::numeric_limits<std::uint64_t>::max();
  return lhs > limit - rhs ? limit : lhs + rhs;
}

/// Bytes an Arrow child occupies, by format string. Used for two different
/// purposes with different column sets: the resource cap counts every child
/// (the unprojected vector included, because Lance materialized it whether we
/// want it or not), while the size estimate counts only imported children.
std::uint64_t child_bytes(ArrowSchema const& schema, ArrowArray const& array) noexcept
{
  auto const rows = array.length < 0 ? 0 : static_cast<std::uint64_t>(array.length);
  if (rows == 0 || schema.format == nullptr) { return 0; }

  auto const validity = (array.null_count != 0) ? (rows + 7) / 8 : 0;
  std::string_view const format{schema.format};

  auto fixed = [&](std::uint64_t width) { return saturating_add(rows * width, validity); };

  if (format == "b") { return fixed(1); }
  if (format == "c" || format == "C") { return fixed(1); }
  if (format == "s" || format == "S") { return fixed(2); }
  if (format == "i" || format == "I" || format == "f" || format == "d") { return fixed(4); }
  if (format == "l" || format == "L" || format == "g") { return fixed(8); }
  if (format.starts_with("ts")) { return fixed(8); }

  // utf8 / large_utf8: offsets plus the character payload, read from the final
  // offset so a sliced or partially-filled buffer is still accounted honestly.
  if (format == "u" || format == "U") {
    auto const wide         = (format == "U");
    auto const offset_width = wide ? 8U : 4U;
    std::uint64_t chars     = 0;
    if (array.n_buffers >= 3 && array.buffers[1] != nullptr) {
      auto const base = static_cast<std::uint64_t>(array.offset);
      if (wide) {
        auto const* offsets = static_cast<std::int64_t const*>(array.buffers[1]);
        chars               = static_cast<std::uint64_t>(offsets[base + rows] - offsets[base]);
      } else {
        auto const* offsets = static_cast<std::int32_t const*>(array.buffers[1]);
        chars               = static_cast<std::uint64_t>(offsets[base + rows] - offsets[base]);
      }
    }
    return saturating_add(saturating_add((rows + 1) * offset_width, chars), validity);
  }

  // Fixed-size list, e.g. "+w:768" — the vector column. Its child holds the
  // flat values, so the width comes from the list size times the child's own
  // per-row cost.
  if (format.starts_with("+w:")) {
    auto const list_size = std::strtoull(schema.format + 3, nullptr, 10);
    if (list_size == 0 || schema.n_children < 1 || array.n_children < 1) { return validity; }
    auto const per_row = child_bytes(*schema.children[0], *array.children[0]);
    auto const values =
      array.children[0]->length > 0
        ? per_row * rows * list_size / static_cast<std::uint64_t>(array.children[0]->length)
        : 0;
    return saturating_add(values, validity);
  }

  return validity;
}

}  // namespace

struct lance_split_producer::impl {
  struct item {
    std::unique_ptr<sirius::op::scan::lance_scan_info> split;
    std::exception_ptr error;
    bool eof{false};
  };

  impl(std::shared_ptr<lance_ffi_api> api_in,
       lance_open_spec spec_in,
       schema_expectation expect_in,
       config cfg_in)
    : api(std::move(api_in)), spec(std::move(spec_in)), expect(std::move(expect_in)), cfg(cfg_in)
  {
    if (!api) { throw std::invalid_argument("lance_split_producer requires a Lance FFI API"); }
    if (cfg.queue_depth == 0) { cfg.queue_depth = 1; }
  }

  std::shared_ptr<lance_ffi_api> api;
  lance_open_spec spec;
  schema_expectation expect;
  config cfg;

  std::mutex mutex;
  std::condition_variable space_available;  ///< producer waits here when the queue is full
  std::condition_variable item_available;   ///< claim waits here when the queue is empty
  std::deque<item> queue;
  bool stop_requested{false};

  std::atomic<std::uint64_t> batches_produced{0};
  std::atomic<std::uint64_t> bytes_produced{0};
  std::atomic<std::uint64_t> queue_high_water{0};
  std::atomic<bool> byte_cap_tripped{false};
  std::atomic<bool> eof_reached{false};
  std::atomic<bool> claimed_eof{false};

  std::thread thread;
  std::once_flag stop_once;

  /// Blocks until the queue has room. Returns false once a stop was requested,
  /// which tells the producer loop to drop the batch and unwind.
  bool push(item value)
  {
    std::unique_lock lock(mutex);
    space_available.wait(lock, [&] { return stop_requested || queue.size() < cfg.queue_depth; });
    if (stop_requested) { return false; }
    queue.push_back(std::move(value));
    queue_high_water.store(std::max(queue_high_water.load(std::memory_order_relaxed),
                                    static_cast<std::uint64_t>(queue.size())),
                           std::memory_order_relaxed);
    lock.unlock();
    item_available.notify_one();
    return true;
  }

  /// Terminal markers must land even after a stop, otherwise a consumer still
  /// sitting in claim() would never be released.
  void push_terminal(item value)
  {
    {
      std::lock_guard lock(mutex);
      queue.push_back(std::move(value));
    }
    item_available.notify_one();
  }

  void validate_schema(ArrowSchema const& schema) const
  {
    for (std::size_t i = 0; i < expect.kept_child_indices.size(); ++i) {
      auto const child_index = expect.kept_child_indices[i];
      if (static_cast<std::int64_t>(child_index) >= schema.n_children) {
        throw std::runtime_error("Lance batch schema drift: expected column '" +
                                 expect.kept_names[i] + "' at child index " +
                                 std::to_string(child_index) + ", but the batch has only " +
                                 std::to_string(schema.n_children) + " children");
      }
      auto const& child = *schema.children[child_index];
      auto const actual_format =
        child.format != nullptr ? std::string{child.format} : std::string{};
      if (actual_format != expect.kept_formats[i]) {
        throw std::runtime_error("Lance batch schema drift: column '" + expect.kept_names[i] +
                                 "' changed format from '" + expect.kept_formats[i] + "' to '" +
                                 actual_format + "'");
      }
      auto const actual_name = child.name != nullptr ? std::string{child.name} : std::string{};
      if (actual_name != expect.kept_names[i]) {
        throw std::runtime_error("Lance batch schema drift: column at child index " +
                                 std::to_string(child_index) + " was named '" +
                                 expect.kept_names[i] + "' at bind time but is now '" +
                                 actual_name + "'");
      }
    }
  }

  /// Decoded device bytes for the imported columns only — the vector column is
  /// excluded because it never reaches cudf.
  std::uint64_t imported_bytes(lance_owning_arrow_batch const& batch) const noexcept
  {
    std::uint64_t total = 0;
    for (auto const child_index : expect.kept_child_indices) {
      if (static_cast<std::int64_t>(child_index) >= batch.schema().n_children ||
          static_cast<std::int64_t>(child_index) >= batch.array().n_children) {
        continue;
      }
      total = saturating_add(
        total,
        child_bytes(*batch.schema().children[child_index], *batch.array().children[child_index]));
    }
    return total;
  }

  /// Every column, imported or not. This is what the resource cap bounds.
  std::uint64_t full_arrow_bytes(lance_owning_arrow_batch const& batch) const noexcept
  {
    std::uint64_t total   = 0;
    auto const& schema    = batch.schema();
    auto const& array     = batch.array();
    auto const n_children = std::min(schema.n_children, array.n_children);
    for (std::int64_t i = 0; i < n_children; ++i) {
      total = saturating_add(total, child_bytes(*schema.children[i], *array.children[i]));
    }
    return total;
  }

  void run()
  {
    void* dataset = nullptr;
    void* stream  = nullptr;
    try {
      lance_error err;
      // Opened here, on this thread, at execution time: that is what makes a
      // re-executed prepared statement observe the dataset's current version.
      dataset = api->dataset_open(spec.uri, spec.storage_options, err);
      if (dataset == nullptr) {
        throw std::runtime_error("Lance dataset open failed for '" + spec.uri +
                                 "': " + err.message);
      }

      stream = api->knn_stream_open(dataset, spec.knn, err);
      if (stream == nullptr) {
        throw std::runtime_error("Lance KNN stream open failed for '" + spec.uri +
                                 "': " + err.message);
      }

      for (;;) {
        {
          std::lock_guard lock(mutex);
          if (stop_requested) { break; }
        }

        ArrowArray array{};
        ArrowSchema schema{};
        lance_error next_err;
        auto const rc = api->stream_next(stream, array, schema, next_err);
        if (rc == 1) { break; }
        if (rc != 0) { throw std::runtime_error("Lance KNN stream failed: " + next_err.message); }

        // From here the batch is owned; every exit path releases it.
        lance_owning_arrow_batch batch(std::move(array), std::move(schema));

        // Account before enqueueing so the cap bounds what we have accepted,
        // and trip on the batch that crosses it rather than one batch late.
        auto const full  = full_arrow_bytes(batch);
        auto const total = saturating_add(bytes_produced.load(std::memory_order_relaxed), full);
        bytes_produced.store(total, std::memory_order_relaxed);
        if (cfg.max_arrow_bytes != 0 && total > cfg.max_arrow_bytes) {
          byte_cap_tripped.store(true, std::memory_order_relaxed);
          batch.release();
          throw std::runtime_error(
            "Lance scan exceeded its Arrow byte cap: accepted " + std::to_string(total) +
            " bytes (including the unprojected vector column) against a limit of " +
            std::to_string(cfg.max_arrow_bytes) +
            "; raise sirius_lance_max_arrow_bytes or lower k");
        }

        validate_schema(batch.schema());

        auto split           = std::make_unique<sirius::op::scan::lance_scan_info>();
        split->decoded_bytes = static_cast<std::size_t>(imported_bytes(batch));
        split->batch         = std::move(batch);

        if (!push(item{std::move(split), nullptr, false})) { break; }
        batches_produced.fetch_add(1, std::memory_order_relaxed);
      }
      eof_reached.store(true, std::memory_order_relaxed);
    } catch (...) {
      push_terminal(item{nullptr, std::current_exception(), false});
    }

    // Stream before dataset: the stream borrows the dataset handle.
    if (stream != nullptr) { api->stream_close(stream); }
    if (dataset != nullptr) { api->dataset_close(dataset); }
    push_terminal(item{nullptr, nullptr, true});
  }
};

lance_split_producer::lance_split_producer(std::shared_ptr<lance_ffi_api> api,
                                           lance_open_spec spec,
                                           schema_expectation expect,
                                           config cfg)
  : _impl(std::make_unique<impl>(std::move(api), std::move(spec), std::move(expect), cfg))
{
  // Constructing a producer starts the scan. Deferring work for a plan that may
  // never execute is the ingestible's job, which creates its producer lazily.
  _impl->thread = std::thread([state = _impl.get()] { state->run(); });
}

lance_split_producer::~lance_split_producer() { stop(); }

sirius::op::scan::gpu_ingestible::metadata_scan_task_t lance_split_producer::claim()
{
  impl::item next;
  {
    std::unique_lock lock(_impl->mutex);
    _impl->item_available.wait(lock, [&] { return !_impl->queue.empty(); });
    next = std::move(_impl->queue.front());
    _impl->queue.pop_front();
  }
  _impl->space_available.notify_one();

  if (next.eof) {
    _impl->claimed_eof.store(true, std::memory_order_release);
    return {};
  }

  if (next.error) {
    // Routed through the returned callable rather than thrown here: the scan
    // driver catches callable exceptions and closes the connector with them,
    // whereas an exception from the claim itself escapes query setup.
    return [error = next.error]() -> std::unique_ptr<sirius::op::scan::scan_info> {
      std::rethrow_exception(error);
    };
  }

  // std::function requires a copyable callable, so the moved-in split rides
  // inside a shared holder.
  auto holder =
    std::make_shared<std::unique_ptr<sirius::op::scan::lance_scan_info>>(std::move(next.split));
  return [holder]() -> std::unique_ptr<sirius::op::scan::scan_info> { return std::move(*holder); };
}

bool lance_split_producer::eof_claimed() const noexcept
{
  return _impl->claimed_eof.load(std::memory_order_acquire);
}

void lance_split_producer::stop() noexcept
{
  std::call_once(_impl->stop_once, [this] {
    {
      std::lock_guard lock(_impl->mutex);
      _impl->stop_requested = true;
    }
    // Wake the producer if it is parked on a full queue. An in-flight FFI call
    // cannot be interrupted — the Lance surface exposes no cancellation — so the
    // join below waits for it, bounded only by the storage-option timeouts.
    _impl->space_available.notify_all();
    _impl->item_available.notify_all();
    if (_impl->thread.joinable()) { _impl->thread.join(); }
  });
}

lance_split_producer::stats lance_split_producer::snapshot() const noexcept
{
  return stats{
    .batches_produced = _impl->batches_produced.load(std::memory_order_relaxed),
    .bytes_produced   = _impl->bytes_produced.load(std::memory_order_relaxed),
    .queue_high_water = _impl->queue_high_water.load(std::memory_order_relaxed),
    .byte_cap_tripped = _impl->byte_cap_tripped.load(std::memory_order_relaxed),
    .eof_reached      = _impl->eof_reached.load(std::memory_order_relaxed),
  };
}

}  // namespace sirius::lance
