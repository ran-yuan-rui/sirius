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

// duckdb — MUST come before any other Arrow-adjacent header. This file has no
// #pragma once; its whole-file guard is `#ifndef ARROW_FLAG_DICTIONARY_ORDERED`,
// so anything that defines that macro first makes ArrowSchema/ArrowArray vanish.
#include <duckdb/common/arrow/arrow.hpp>

// standard library
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::lance {

//===----------------------------------------------------------------------===//
// lance_error
//===----------------------------------------------------------------------===//
/**
 * @brief Error detail transported out of the FFI boundary.
 *
 * The underlying Lance FFI keeps its last error in a thread-local slot whose
 * message read is destructive, so the shim reads code-then-message on the same
 * thread that observed the failure and hands the result back through this
 * caller-owned struct. Callers never touch thread-local state.
 */
struct lance_error {
  std::int32_t code{0};
  std::string message;

  [[nodiscard]] bool ok() const noexcept { return code == 0 && message.empty(); }
  void clear() noexcept
  {
    code = 0;
    message.clear();
  }
};

//===----------------------------------------------------------------------===//
// knn_params
//===----------------------------------------------------------------------===//
/**
 * @brief Vector-search parameters resolved at bind time.
 *
 * @c query holds the probe vector; its size must equal the fixed-size-list
 * width of @c vector_column (validated during bind). @c k is always positive:
 * a non-positive value is rejected before this struct is built.
 */
struct knn_params {
  std::string vector_column;
  std::vector<float> query;
  std::uint64_t k{10};
  std::uint64_t nprobes{0};
  std::uint64_t refine_factor{0};
  bool prefilter{false};
  bool use_index{true};
};

/// Non-secret storage options forwarded verbatim to Lance's own object store
/// (endpoint, region, timeouts, ...). Credential-bearing keys are rejected at
/// bind time; credentials come from the ambient AWS provider chain instead.
using storage_option_list = std::vector<std::pair<std::string, std::string>>;

//===----------------------------------------------------------------------===//
// lance_ffi_api
//===----------------------------------------------------------------------===//
/**
 * @brief Abstract Lance FFI surface consumed by the Sirius Lance scan source.
 *
 * One production implementation wraps the pinned `lance_duckdb_ffi` staticlib
 * and is only linked when @c SIRIUS_ENABLE_LANCE_KNN is ON; tests substitute
 * fakes through @ref scoped_ffi_api_override. Everything above this interface
 * (bind, planner seam, producer, ingestible) compiles and is exercised without
 * the Rust dependency.
 *
 * Threading contract, guaranteed by the production implementation and assumed
 * by every caller:
 * - No raw `lance_*` symbol ever executes on the calling thread. Control calls
 *   (open/close/count/schema) run on the shim's dedicated control thread;
 *   stream calls run on the owning producer thread. A host that embeds DuckDB
 *   inside its own async runtime therefore cannot trip the FFI's internal
 *   `block_on` abort path.
 * - At most one thread at a time may drive a given stream handle.
 * - Every fallible call reports through its @c lance_error out-parameter; the
 *   thread-local slot behind it is read in place, never by the caller.
 */
class lance_ffi_api {
 public:
  virtual ~lance_ffi_api() = default;

  lance_ffi_api(lance_ffi_api const&)            = delete;
  lance_ffi_api& operator=(lance_ffi_api const&) = delete;
  lance_ffi_api(lance_ffi_api&&)                 = delete;
  lance_ffi_api& operator=(lance_ffi_api&&)      = delete;

  /// Opens a dataset at its current version. Returns nullptr and fills @p err on
  /// failure. The returned handle pins that version until @ref dataset_close.
  virtual void* dataset_open(std::string_view uri,
                             storage_option_list const& storage_options,
                             lance_error& err) = 0;

  virtual void dataset_close(void* dataset) noexcept = 0;

  /// Cheap liveness/validation probe. Returns a negative value on failure.
  virtual std::int64_t dataset_count_rows(void* dataset, lance_error& err) = 0;

  /// Fills @p out with the schema a KNN stream for @p p would produce. The
  /// caller owns @p out and must invoke its release callback. Note the Lance
  /// KNN surface takes no projection list: the schema carries every dataset
  /// column plus `_distance`, including the vector column itself.
  virtual bool knn_schema(void* dataset,
                          knn_params const& p,
                          ArrowSchema& out,
                          lance_error& err) = 0;

  virtual void* knn_stream_open(void* dataset, knn_params const& p, lance_error& err) = 0;

  /**
   * @brief Advances the stream by one batch.
   *
   * @return 0 when a batch was produced (@p out_array and @p out_schema are
   *         filled and owned by the caller, which must release both), 1 at
   *         end of stream, -1 on error (@p err is filled).
   *
   * Fuses next + export + handle-free so a raw batch handle never escapes the
   * implementation and the error slot is consumed on the producing thread.
   */
  virtual std::int32_t stream_next(void* stream,
                                   ArrowArray& out_array,
                                   ArrowSchema& out_schema,
                                   lance_error& err) = 0;

  virtual void stream_close(void* stream) noexcept = 0;

 protected:
  lance_ffi_api() = default;
};

//===----------------------------------------------------------------------===//
// injection seam
//===----------------------------------------------------------------------===//
using lance_ffi_api_factory = std::function<std::shared_ptr<lance_ffi_api>()>;

/**
 * @brief Installs an API factory for the guard's lifetime; restores the
 *        previous one on destruction.
 *
 * This is the only supported way to substitute a fake: tests install a guard,
 * run SQL, and let the guard unwind. Bind data holds a @c shared_ptr to the
 * product, so a plan bound while the guard was live keeps using its API even
 * after the guard is gone — which is what makes repeated EXECUTE of a prepared
 * statement observable to the test.
 */
class scoped_ffi_api_override {
 public:
  explicit scoped_ffi_api_override(lance_ffi_api_factory factory);
  ~scoped_ffi_api_override();

  scoped_ffi_api_override(scoped_ffi_api_override const&)            = delete;
  scoped_ffi_api_override& operator=(scoped_ffi_api_override const&) = delete;
  scoped_ffi_api_override(scoped_ffi_api_override&&)                 = delete;
  scoped_ffi_api_override& operator=(scoped_ffi_api_override&&)      = delete;

 private:
  lance_ffi_api_factory _previous;
};

/**
 * @brief Bind-time accessor for the active API.
 *
 * Returns the installed factory's product when a @ref scoped_ffi_api_override
 * is live, otherwise the production implementation. Throws
 * @c std::runtime_error when no implementation is available — the build was
 * configured without @c SIRIUS_ENABLE_LANCE_KNN and no override is installed.
 * The bind callback converts that into a user-facing binder error.
 */
[[nodiscard]] std::shared_ptr<lance_ffi_api> acquire_lance_ffi_api();

/// Whether a call to @ref acquire_lance_ffi_api would succeed. Lets the bind
/// callback produce a precise diagnostic instead of a caught exception.
[[nodiscard]] bool lance_ffi_api_available() noexcept;

}  // namespace sirius::lance
