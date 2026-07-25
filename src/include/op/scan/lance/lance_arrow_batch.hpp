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

// sirius — pulls in the Arrow C data interface (see the include-order note there)
#include <lance_shim/lance_ffi_api.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <cstdint>
#include <memory>

namespace sirius::lance {

//===----------------------------------------------------------------------===//
// lance_owning_arrow_batch
//===----------------------------------------------------------------------===//
/**
 * @brief Move-only RAII owner of one exported Arrow record batch.
 *
 * The Lance FFI transfers ownership of an @c ArrowArray / @c ArrowSchema pair
 * to the caller, which must invoke each release callback exactly once. This
 * type makes that automatic and idempotent.
 *
 * Destruction is host-only: it frees host buffers and makes no CUDA calls. That
 * matters because the scan driver destroys undrained splits on the query
 * teardown thread, which carries no CUDA-context guarantees.
 *
 * Lifetime rule for GPU import: cudf copies host buffers asynchronously on the
 * caller's stream and does not synchronize before returning, so a batch must
 * stay alive until that stream has run the copies. Enforcement lives in
 * @ref arrow_batch_retirer rather than here.
 */
class lance_owning_arrow_batch {
 public:
  lance_owning_arrow_batch() noexcept = default;

  /// Adopts an array/schema pair already filled by @c lance_ffi_api::stream_next.
  lance_owning_arrow_batch(ArrowArray&& array, ArrowSchema&& schema) noexcept;

  ~lance_owning_arrow_batch();

  lance_owning_arrow_batch(lance_owning_arrow_batch&& other) noexcept;
  lance_owning_arrow_batch& operator=(lance_owning_arrow_batch&& other) noexcept;

  lance_owning_arrow_batch(lance_owning_arrow_batch const&)            = delete;
  lance_owning_arrow_batch& operator=(lance_owning_arrow_batch const&) = delete;

  [[nodiscard]] ArrowArray& array() noexcept { return _array; }
  [[nodiscard]] ArrowArray const& array() const noexcept { return _array; }
  [[nodiscard]] ArrowSchema& schema() noexcept { return _schema; }
  [[nodiscard]] ArrowSchema const& schema() const noexcept { return _schema; }

  /// Whether both release callbacks are still present, i.e. the batch owns data.
  [[nodiscard]] bool valid() const noexcept;

  /// Row count of the top-level struct array; 0 when not @ref valid.
  [[nodiscard]] std::int64_t num_rows() const noexcept;

  /// Releases schema then array. Idempotent; safe on a moved-from or empty batch.
  void release() noexcept;

 private:
  ArrowArray _array{};
  ArrowSchema _schema{};
};

//===----------------------------------------------------------------------===//
// arrow_batch_retirer
//===----------------------------------------------------------------------===//
/**
 * @brief Policy seam deciding WHEN a batch's host buffers may be freed
 *        relative to the CUDA stream that imported them.
 *
 * The v1 policy synchronizes the task stream and then drops the batch. A later
 * event-based policy can record an event and free on completion without any
 * change to callers, the producer, or the scan_info contract.
 *
 * Implementations must be callable from a GPU task thread and must leave the
 * batch released once @ref retire returns or throws.
 */
class arrow_batch_retirer {
 public:
  virtual ~arrow_batch_retirer() = default;

  arrow_batch_retirer(arrow_batch_retirer const&)            = delete;
  arrow_batch_retirer& operator=(arrow_batch_retirer const&) = delete;
  arrow_batch_retirer(arrow_batch_retirer&&)                 = delete;
  arrow_batch_retirer& operator=(arrow_batch_retirer&&)      = delete;

  virtual void retire(lance_owning_arrow_batch&& batch, rmm::cuda_stream_view stream) = 0;

 protected:
  arrow_batch_retirer() = default;
};

/// v1 policy: synchronize @p stream, then release the batch. Conservative and
/// correct; the seam exists so a deferred policy can replace it later.
[[nodiscard]] std::unique_ptr<arrow_batch_retirer> make_synchronizing_retirer();

/// Factory stored on the table info so tests can substitute a recording or
/// deferring policy without reaching into ingestible internals.
using arrow_batch_retirer_factory = std::function<std::unique_ptr<arrow_batch_retirer>()>;

}  // namespace sirius::lance
