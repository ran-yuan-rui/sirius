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

#include <data/gpu_parquet_representation.hpp>

#include <rmm/device_buffer.hpp>

#include <stdexcept>
#include <utility>

namespace sirius {

gpu_parquet_representation::gpu_parquet_representation(
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
  std::shared_ptr<translated_expression> filter_expression,
  std::vector<std::size_t> post_filter_projection_ids)
  : idata_representation(memory_space),
    _column_chunks(std::move(column_chunks)),
    _parquet_reader(std::move(parquet_reader)),
    _reader_options(std::move(reader_options)),
    _row_group_indices(std::move(row_group_indices)),
    _column_chunk_byte_ranges(std::move(column_chunk_byte_ranges)),
    _size_in_bytes(size_in_bytes),
    _uncompressed_size_in_bytes(uncompressed_size_in_bytes),
    _file_size(file_size),
    _fallback_datasource(std::move(fallback_datasource)),
    _filter_expression(std::move(filter_expression)),
    _post_filter_projection_ids(std::move(post_filter_projection_ids))
{
}

std::size_t gpu_parquet_representation::get_size_in_bytes() const { return _size_in_bytes; }

std::size_t gpu_parquet_representation::get_uncompressed_data_size_in_bytes() const
{
  return _uncompressed_size_in_bytes;
}

std::unique_ptr<cucascade::idata_representation> gpu_parquet_representation::clone(
  rmm::cuda_stream_view stream)
{
  rmm::device_buffer cloned_buf(_column_chunks, stream);
  auto cloned_reader =
    std::make_unique<hybrid_scan_reader>(_parquet_reader->parquet_metadata(), _reader_options);

  auto result = std::make_unique<gpu_parquet_representation>(get_memory_space(),
                                                             std::move(cloned_buf),
                                                             std::move(cloned_reader),
                                                             _reader_options,
                                                             _row_group_indices,
                                                             _column_chunk_byte_ranges,
                                                             _size_in_bytes,
                                                             _uncompressed_size_in_bytes,
                                                             _file_size,
                                                             _fallback_datasource,
                                                             _filter_expression,
                                                             _post_filter_projection_ids);
  if (_post_convert_fn) { result->set_post_convert_fn(_post_convert_fn); }
  if (!_data_file_path.empty()) { result->set_data_file_path(_data_file_path); }
  return result;
}

rmm::device_buffer const& gpu_parquet_representation::get_column_chunks() const
{
  return _column_chunks;
}

rmm::device_buffer& gpu_parquet_representation::get_column_chunks() { return _column_chunks; }

std::shared_ptr<gpu_parquet_representation::hybrid_scan_reader>
gpu_parquet_representation::get_parquet_reader() const
{
  return _parquet_reader;
}

cudf::io::parquet_reader_options const& gpu_parquet_representation::get_reader_options() const
{
  return _reader_options;
}

std::vector<cudf::size_type> const& gpu_parquet_representation::get_row_group_indices() const
{
  return _row_group_indices;
}

std::vector<cudf::io::text::byte_range_info> const&
gpu_parquet_representation::get_column_chunk_byte_ranges() const
{
  return _column_chunk_byte_ranges;
}

std::size_t gpu_parquet_representation::get_file_size() const { return _file_size; }

std::shared_ptr<cudf::io::datasource> const&
gpu_parquet_representation::get_fallback_datasource() const
{
  return _fallback_datasource;
}

std::shared_ptr<gpu_parquet_representation::translated_expression> const&
gpu_parquet_representation::get_filter_expression() const
{
  return _filter_expression;
}

std::vector<std::size_t> const& gpu_parquet_representation::get_post_filter_projection_ids() const
{
  return _post_filter_projection_ids;
}

void gpu_parquet_representation::set_post_convert_fn(post_convert_fn_t fn)
{
  _post_convert_fn = std::move(fn);
}

bool gpu_parquet_representation::has_post_convert_fn() const
{
  return static_cast<bool>(_post_convert_fn);
}

std::unique_ptr<cudf::table> gpu_parquet_representation::apply_post_convert(
  std::unique_ptr<cudf::table> tbl, rmm::cuda_stream_view stream)
{
  if (!_post_convert_fn) { return tbl; }
  auto first_row = compute_first_row_offset();
  return _post_convert_fn(std::move(tbl), _data_file_path, first_row, stream);
}

void gpu_parquet_representation::set_data_file_path(std::string path)
{
  _data_file_path = std::move(path);
}

std::string const& gpu_parquet_representation::get_data_file_path() const
{
  return _data_file_path;
}

int64_t gpu_parquet_representation::compute_first_row_offset() const
{
  if (_row_group_indices.empty() || !_parquet_reader) { return 0; }

  auto const& metadata = _parquet_reader->parquet_metadata();
  int64_t offset       = 0;
  auto first_rg        = _row_group_indices.front();
  for (cudf::size_type rg = 0; rg < first_rg; ++rg) {
    offset += metadata.row_groups[rg].num_rows;
  }
  return offset;
}

}  // namespace sirius
