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

// sirius
#include <lance_shim/lance_ffi_api.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/lance/lance_arrow_batch.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sirius::lance {

//===----------------------------------------------------------------------===//
// lance_open_spec
//===----------------------------------------------------------------------===//
/**
 * @brief Everything needed to reopen the dataset and start one KNN stream.
 *
 * Held by value so each execution is self-contained: the producer reopens the
 * dataset when it starts, which is what makes a repeated EXECUTE of a prepared
 * statement observe the dataset's then-current version. Within one execution
 * the version is fixed for as long as the handle is held.
 */
struct lance_open_spec {
  std::string uri;
  storage_option_list storage_options;
  knn_params knn;
};

//===----------------------------------------------------------------------===//
// schema_expectation
//===----------------------------------------------------------------------===//
/**
 * @brief Bind-time schema contract the producer re-checks on every batch.
 *
 * @c kept_child_indices lists, in emission order, the top-level children the
 * scan imports. The vector column is deliberately absent: the Lance KNN surface
 * has no projection list, so the vector always arrives in the host Arrow batch,
 * and dropping it by index here is what keeps it out of cudf and off the device.
 *
 * @c kept_formats and @c kept_names are the per-child Arrow format strings and
 * names captured at bind time. A mismatch means the dataset changed between
 * bind and execute and is reported through the stream error channel.
 */
struct schema_expectation {
  std::vector<std::size_t> kept_child_indices;
  std::vector<std::string> kept_formats;
  std::vector<std::string> kept_names;

  [[nodiscard]] std::size_t kept_count() const noexcept { return kept_child_indices.size(); }
};

//===----------------------------------------------------------------------===//
// lance_split_producer
//===----------------------------------------------------------------------===//
/**
 * @brief Single-threaded Lance batch producer feeding one bounded queue.
 *
 * One dedicated std::thread owns the dataset handle, the stream, and every FFI
 * call for that stream, which keeps the FFI's one-thread-per-stream rule and
 * its thread-local error slot correct by construction, and keeps all Lance work
 * off both the Sirius worker pool and any host async runtime.
 *
 * The queue bounds how far the producer may run ahead of @ref claim; it does
 * not provide end-to-end backpressure, because the scan driver's own connector
 * is unbounded. Total host residency is bounded instead by @c max_arrow_bytes
 * (see @ref config) plus the bind-time cap on @c k.
 */
class lance_split_producer {
 public:
  struct config {
    /// Producer run-ahead bound, in batches.
    std::size_t queue_depth{4};
    /// Cumulative cap on exported Arrow bytes, counting every column including
    /// the unprojected vector column. Accounted before a batch is enqueued,
    /// with saturating addition. 0 disables the cap (tests only).
    std::uint64_t max_arrow_bytes{0};
  };

  /// Observation surface for tests and, later, telemetry. Cheap snapshot; safe
  /// to call from any thread at any time.
  struct stats {
    std::uint64_t batches_produced{0};
    /// Full exported Arrow bytes, including the vector column.
    std::uint64_t bytes_produced{0};
    std::uint64_t queue_high_water{0};
    bool byte_cap_tripped{false};
    bool eof_reached{false};
  };

  lance_split_producer(std::shared_ptr<lance_ffi_api> api,
                       lance_open_spec spec,
                       schema_expectation expect,
                       config cfg);

  /// Stops and joins the producer thread.
  ~lance_split_producer();

  lance_split_producer(lance_split_producer const&)            = delete;
  lance_split_producer& operator=(lance_split_producer const&) = delete;
  lance_split_producer(lance_split_producer&&)                 = delete;
  lance_split_producer& operator=(lance_split_producer&&)      = delete;

  /**
   * @brief Claims the next unit of work, blocking until one is available.
   *
   * Returns either a callable yielding one @c scan_info, a callable that
   * rethrows the producer's error, or nullptr exactly once at end of stream.
   * Blocking here is deliberate: the driver's claim loop spins on a null
   * callable, so a transient "nothing yet" return would busy-wait.
   *
   * Called only from the scan driver's claim thread, which is serialized.
   */
  [[nodiscard]] sirius::op::scan::gpu_ingestible::metadata_scan_task_t claim();

  /// Whether the end-of-stream sentinel has been claimed. Lock-free; polled by
  /// @c gpu_ingestible::has_processed_all_metadata.
  [[nodiscard]] bool eof_claimed() const noexcept;

  /**
   * @brief Requests shutdown and joins the producer thread. Idempotent.
   *
   * Worst-case wait is one in-flight FFI call, which is bounded only by the
   * Lance storage-option timeouts the bind step always installs: an in-flight
   * stream read cannot be interrupted, because the FFI exposes no cancellation.
   */
  void stop() noexcept;

  [[nodiscard]] stats snapshot() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::lance
