/*
 * Copyright 2025, Sirius Contributors.
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
#include <data/host_parquet_representation.hpp>  // for post_convert_fn_t
#include <expression_executor/gpu_expression_translator.hpp>

// cucascade
#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

// standard library
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sirius {

// Re-use post_convert_fn_t from host_parquet_representation.hpp (same type).
// Forward-declared here to avoid circular include; the actual type alias
// is defined in data/host_parquet_representation.hpp.

/**
 * @brief Compressed Parquet column chunks resident in GPU memory.
 *
 * Used by the GPU Direct path (GDS, RDMA) to avoid host-memory staging.
 * The compressed bytes are read directly into an rmm::device_buffer via
 * device_read_async(), then decompressed in-place on GPU by the converter
 * (gpu_parquet_representation -> gpu_table_representation).
 *
 * Mirrors host_parquet_representation but stores data in GPU memory instead
 * of host fixed-size block allocations.
 */
class gpu_parquet_representation : public cucascade::idata_representation {
  using hybrid_scan_reader    = cudf::io::parquet::experimental::hybrid_scan_reader;
  using translated_expression = gpu_expression_translator::translated_expression;

 public:
  gpu_parquet_representation(
    cucascade::memory::memory_space& memory_space,
    rmm::device_buffer column_chunks,
    std::shared_ptr<hybrid_scan_reader> parquet_reader,
    cudf::io::parquet_reader_options reader_options,
    std::vector<cudf::size_type> row_group_indices,
    std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges,
    std::size_t size_in_bytes,
    std::size_t uncompressed_size_in_bytes,
    std::size_t file_size,
    std::shared_ptr<cudf::io::datasource> fallback_datasource,
    std::shared_ptr<translated_expression> filter_expression = nullptr,
    std::vector<std::size_t> post_filter_projection_ids = {});

  [[nodiscard]] std::size_t get_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;
  [[nodiscard]] std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;

  [[nodiscard]] rmm::device_buffer const& get_column_chunks() const;
  [[nodiscard]] rmm::device_buffer& get_column_chunks();

  [[nodiscard]] std::shared_ptr<hybrid_scan_reader> get_parquet_reader() const;
  [[nodiscard]] cudf::io::parquet_reader_options const& get_reader_options() const;
  [[nodiscard]] std::vector<cudf::size_type> const& get_row_group_indices() const;
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const&
  get_column_chunk_byte_ranges() const;
  [[nodiscard]] std::size_t get_file_size() const;
  [[nodiscard]] std::shared_ptr<cudf::io::datasource> const& get_fallback_datasource() const;
  [[nodiscard]] std::shared_ptr<translated_expression> const& get_filter_expression() const;
  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const;

  void set_post_convert_fn(post_convert_fn_t fn);
  [[nodiscard]] bool has_post_convert_fn() const;
  [[nodiscard]] std::unique_ptr<cudf::table> apply_post_convert(
    std::unique_ptr<cudf::table> tbl, rmm::cuda_stream_view stream);

  void set_data_file_path(std::string path);
  [[nodiscard]] std::string const& get_data_file_path() const;
  [[nodiscard]] int64_t compute_first_row_offset() const;

 private:
  rmm::device_buffer _column_chunks;
  std::shared_ptr<hybrid_scan_reader> _parquet_reader;
  cudf::io::parquet_reader_options _reader_options;
  std::vector<cudf::size_type> _row_group_indices;
  std::vector<cudf::io::text::byte_range_info> _column_chunk_byte_ranges;
  std::size_t _size_in_bytes;
  std::size_t _uncompressed_size_in_bytes;
  std::size_t _file_size{0};
  std::shared_ptr<cudf::io::datasource> _fallback_datasource;
  std::shared_ptr<translated_expression> _filter_expression;
  std::vector<std::size_t> _post_filter_projection_ids;
  post_convert_fn_t _post_convert_fn;
  std::string _data_file_path;
};

}  // namespace sirius
