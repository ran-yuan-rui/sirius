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
#include <helper/logical_type.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/lance/lance_arrow_batch.hpp>
#include <op/scan/lance/lance_split_producer.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sirius::op {
class sirius_dynamic_filter_set;
}  // namespace sirius::op

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// lance_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Bind data for a Lance vector-search scan; builds the
 *        @c lance_gpu_ingestible.
 *
 * Carries the FFI handle-maker rather than an open dataset: the producer opens
 * the dataset when an execution starts, so a re-executed prepared statement
 * observes the dataset's then-current version.
 */
class lance_ingestible_table_info : public op::scan::ingestible_table_info {
 public:
  /// Active FFI implementation, captured at bind time. Production or fake; the
  /// scan layer cannot tell the difference.
  std::shared_ptr<sirius::lance::lance_ffi_api> api;

  /// Dataset URI, non-secret storage options, and KNN parameters.
  sirius::lance::lance_open_spec spec;

  /// Bind-time schema contract; also encodes which children are imported and,
  /// by omission, that the vector column never reaches cudf.
  sirius::lance::schema_expectation expect;

  /// Producer queue depth and cumulative Arrow byte cap.
  sirius::lance::lance_split_producer::config producer_config;

  /// Host-batch retirement policy. Defaults to the synchronizing policy; tests
  /// substitute a recording or deferring one to assert release ordering.
  sirius::lance::arrow_batch_retirer_factory retirer_factory;

  /// Output column names in emission order, `_distance` included.
  std::vector<std::string> names;

  /// Output column types, positionally aligned with @c names.
  std::vector<sirius::logical_type> output_types;

  /// Storage indices of the emitted columns, in emission order. Feeds
  /// @c gpu_ingestible::materialized_column_order.
  std::vector<std::size_t> materialized_order;

  /// Required by @c make_gpu_scan_leaf. Always null in v1: the table function
  /// registers with filter pushdown disabled, so no dynamic-filter channel is
  /// wired to this source.
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> sirius_dynamic_filters;

  lance_ingestible_table_info() = default;

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }

  /// The dataset URI as a one-element span (contract-keeping for the base
  /// interface). Lance scans never participate in pinned-cache matching, so
  /// this value is not used for identity.
  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    if (spec.uri.empty()) { return {}; }
    return std::span<std::string const>(&spec.uri, 1);
  }
};

//===----------------------------------------------------------------------===//
// lance_scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief One unit of Lance scan work: exactly one host Arrow record batch.
 *
 * Destruction is host-only and makes no CUDA calls, so the driver may destroy
 * undrained splits on the teardown thread.
 *
 * The batch is @c mutable because @c materialize_metadata_to_table receives its
 * split by const reference yet must hand the batch to the retirer once the
 * import is safely complete. The driver materializes each split at most once.
 */
class lance_scan_info : public op::scan::scan_info {
 public:
  mutable sirius::lance::lance_owning_arrow_batch batch;

  /// Decoded device bytes for the imported columns only (the vector column is
  /// excluded because it is never imported). Computed once by the producer.
  std::size_t decoded_bytes{0};

  /// True for the synthetic split a coalescer emits when the stream produced no
  /// batches at all: zero splits would mean the pipeline-completion signal
  /// never fires. Materializes to a correctly-typed zero-row table.
  bool is_empty_sentinel{false};

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return decoded_bytes; }
};

//===----------------------------------------------------------------------===//
// lance_gpu_ingestible
//===----------------------------------------------------------------------===//
/**
 * @brief Scan source that turns Lance KNN results into cudf tables.
 *
 * Host Arrow batches arrive from a single @c lance_split_producer; each becomes
 * one split, imported column-by-column with @c cudf::from_arrow_column on the
 * task stream. No DuckDB DataChunk is involved on any path.
 */
class lance_gpu_ingestible : public op::scan::gpu_ingestible {
 public:
  explicit lance_gpu_ingestible(std::unique_ptr<lance_ingestible_table_info> info);
  ~lance_gpu_ingestible() override;

  [[nodiscard]] std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;

  [[nodiscard]] bool has_processed_all_metadata() const override;

  /// Ignores the resolver: a Lance scan owns no sirius_datasource, so the split
  /// exposes no fadvise entries and every prefetch site is a no-op.
  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;

  filtered_table materialize_metadata_to_table(const scan_info& info,
                                               const cucascade::memory::memory_space& mem_space,
                                               rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    const cucascade::memory::memory_space& mem_space,
    rmm::cuda_stream_view stream) override;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept override;

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

/// ADL factory required by @c make_gpu_scan_leaf.
[[nodiscard]] std::shared_ptr<gpu_ingestible> make_ingestible(
  std::unique_ptr<lance_ingestible_table_info> info);

}  // namespace sirius::op::scan
