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

// sirius
#include <data/cached_data_representation.hpp>
#include <data/gpu_parquet_representation.hpp>
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>
#include <op/scan/cached_ranges.hpp>
#include <op/scan/prefetched_data_source.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include "cudf/cudf_utils.hpp"

#include <cudf/utilities/span.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace detail {

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 */
std::unique_ptr<cucascade::idata_representation>
convert_host_parquet_to_gpu_with_prefetched_data_source(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_src                         = source.cast<host_parquet_representation>();
  auto const& post_filter_projection_ids = host_src.get_post_filter_projection_ids();

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  // Build a cache_ranges from the packed host blocks and the column-chunk byte-range descriptors.
  // The block pointers are raw std::byte* owned by the allocation; cache_ranges does not take
  // ownership of them (the allocation still lives for the duration of this call).
  auto const& allocation = host_src.get_column_chunks();
  auto block_ptrs =
    std::vector<std::byte*>(allocation->get_blocks().begin(), allocation->get_blocks().end());

  auto ranges =
    std::make_unique<sirius::op::scan::cache_ranges>(host_src.get_column_chunk_byte_ranges(),
                                                     std::move(block_ptrs),
                                                     allocation->block_size(),
                                                     target_memory_space->get_device_id(),
                                                     host_src.get_device_id());  // NUMA node id

  auto data_source = std::make_unique<sirius::op::scan::prefetched_data_source>(
    std::move(ranges), host_src.get_file_size(), host_src.get_fallback_datasource());

  // Point the reader options at our in-memory datasource and call cudf::io::read_parquet.
  auto opts = host_src.get_reader_options();
  opts.set_source(cudf::io::source_info{data_source.get()});
  // set_row_groups expects one inner vector per source; we have a single source.
  opts.set_row_groups({std::vector<cudf::size_type>(host_src.get_row_group_indices().begin(),
                                                    host_src.get_row_group_indices().end())});

  auto [table, md] = cudf::io::read_parquet(opts, stream, mr_ref);

  // Apply the post-convert hook (used by iceberg scan for V2 delete filtering).
  if (host_src.has_post_convert_fn()) {
    table = host_src.apply_post_convert(std::move(table), stream);
  }

  stream.synchronize();

  // Now we need to prune the post-filter columns from the table, if there are any.
  if (!post_filter_projection_ids.empty()) {
    auto columns = table->release();
    std::vector<std::unique_ptr<cudf::column>> projected_columns;
    projected_columns.reserve(post_filter_projection_ids.size());
    for (auto const id : post_filter_projection_ids) {
      projected_columns.push_back(std::move(columns[id]));
    }
    table = std::make_unique<cudf::table>(std::move(projected_columns));
  }

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_parquet_representation to host_parquet_representation (cross-host copy)
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_host_parquet(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /* stream */)
{
  auto& host_src       = source.cast<host_parquet_representation>();
  auto const data_size = host_src.get_size_in_bytes();

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto* mr = target_memory_space
               ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }

  auto const& src_allocation  = host_src.get_column_chunks();
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_block_index      = 0;
  size_t dst_block_offset     = 0;
  size_t const src_block_size = src_allocation->block_size();
  size_t const dst_block_size = dst_allocation->block_size();
  size_t copied               = 0;
  while (copied < data_size) {
    size_t remaining     = data_size - copied;
    size_t src_avail     = src_block_size - src_block_offset;
    size_t dst_avail     = dst_block_size - dst_block_offset;
    size_t bytes_to_copy = std::min({remaining, src_avail, dst_avail});
    auto* src_ptr        = src_allocation->at(src_block_index).data() + src_block_offset;
    auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
    std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    copied += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    dst_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
    if (dst_block_offset == dst_block_size) {
      dst_block_index++;
      dst_block_offset = 0;
    }
  }

  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;
  auto cloned_reader       = std::make_unique<hybrid_scan_reader>(
    host_src.get_parquet_reader()->parquet_metadata(), host_src.get_reader_options());
  auto dst = std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(cloned_reader),
    host_src.get_reader_options(),
    host_src.get_row_group_indices(),
    host_src.get_column_chunk_byte_ranges(),
    data_size,
    host_src.get_uncompressed_data_size_in_bytes(),
    host_src.get_file_size(),
    host_src.get_fallback_datasource(),
    host_src.get_filter_expression(),
    host_src.get_post_filter_projection_ids());
  if (host_src.has_post_convert_fn()) { dst->set_post_convert_fn(host_src.get_post_convert_fn()); }
  if (!host_src.get_data_file_path().empty()) {
    dst->set_data_file_path(host_src.get_data_file_path());
  }
  return dst;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// gpu_parquet_representation -> gpu_table_representation converter
// ---------------------------------------------------------------------------

namespace detail {

/**
 * @brief A cudf::io::datasource that serves reads from a GPU-resident device buffer.
 *
 * Used by the gpu_parquet -> gpu_table converter to feed the already-GPU-resident
 * compressed Parquet bytes to cudf::io::read_parquet() for in-place decompression.
 */
class device_buffer_datasource : public cudf::io::datasource {
 public:
  device_buffer_datasource(void const* d_data, std::size_t size) : _d_data(d_data), _size(size) {}

  [[nodiscard]] bool supports_device_read() const override { return true; }
  [[nodiscard]] bool is_device_read_preferred(size_t) const override { return true; }

  std::unique_ptr<buffer> device_read(size_t offset, size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    // Return a non-owning buffer pointing into the existing device memory.
    class non_owning_device_buffer : public buffer {
     public:
      non_owning_device_buffer(uint8_t const* ptr, size_t sz) : _ptr(ptr), _sz(sz) {}
      [[nodiscard]] size_t size() const override { return _sz; }
      [[nodiscard]] uint8_t const* data() const override { return _ptr; }

     private:
      uint8_t const* _ptr;
      size_t _sz;
    };

    auto const* ptr = static_cast<uint8_t const*>(_d_data) + offset;
    return std::make_unique<non_owning_device_buffer>(ptr, size);
  }

  size_t device_read(size_t offset, size_t size, uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    auto const* src = static_cast<uint8_t const*>(_d_data) + offset;
    RMM_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream.value()));
    stream.synchronize();
    return size;
  }

  std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    auto bytes = device_read(offset, size, dst, stream);
    std::promise<size_t> p;
    p.set_value(bytes);
    return p.get_future();
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const* src = static_cast<uint8_t const*>(_d_data) + offset;
    RMM_CUDA_TRY(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return size;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto buf = std::make_unique<std::vector<uint8_t>>(size);
    host_read(offset, size, buf->data());
    class owning_buffer : public buffer {
     public:
      owning_buffer(std::unique_ptr<std::vector<uint8_t>> data, size_t len)
        : _data(std::move(data)), _len(len)
      {
      }
      [[nodiscard]] size_t size() const override { return _len; }
      [[nodiscard]] uint8_t const* data() const override { return _data->data(); }

     private:
      std::unique_ptr<std::vector<uint8_t>> _data;
      size_t _len;
    };
    return std::make_unique<owning_buffer>(std::move(buf), size);
  }

  [[nodiscard]] size_t size() const override { return _size; }

 private:
  void const* _d_data;
  std::size_t _size;
};

std::unique_ptr<cucascade::idata_representation> convert_gpu_parquet_to_gpu_table(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& gpu_src                          = source.cast<gpu_parquet_representation>();
  auto const& post_filter_projection_ids = gpu_src.get_post_filter_projection_ids();

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  // Wrap the GPU-resident compressed bytes in a datasource.
  auto data_source = std::make_unique<device_buffer_datasource>(gpu_src.get_column_chunks().data(),
                                                                gpu_src.get_file_size());

  auto opts = gpu_src.get_reader_options();
  opts.set_source(cudf::io::source_info{data_source.get()});
  opts.set_row_groups({std::vector<cudf::size_type>(gpu_src.get_row_group_indices().begin(),
                                                    gpu_src.get_row_group_indices().end())});

  auto [table, md] = cudf::io::read_parquet(opts, stream, mr_ref);

  if (gpu_src.has_post_convert_fn()) {
    table = gpu_src.apply_post_convert(std::move(table), stream);
  }

  stream.synchronize();

  if (!post_filter_projection_ids.empty()) {
    auto columns = table->release();
    std::vector<std::unique_ptr<cudf::column>> projected_columns;
    projected_columns.reserve(post_filter_projection_ids.size());
    for (auto const id : post_filter_projection_ids) {
      projected_columns.push_back(std::move(columns[id]));
    }
    table = std::make_unique<cudf::table>(std::move(projected_columns));
  }

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

}  // namespace detail

void register_parquet_converters(cucascade::representation_converter_registry& registry)
{
  // HOST Parquet -> GPU
  if (!registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_parquet_representation, cucascade::gpu_table_representation>(
      detail::convert_host_parquet_to_gpu_with_prefetched_data_source);
  }

  // GPU Parquet -> GPU Table (in-place GPU decompression for GPU Direct path)
  if (!registry
         .has_converter<gpu_parquet_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<gpu_parquet_representation, cucascade::gpu_table_representation>(
      detail::convert_gpu_parquet_to_gpu_table);
  }

  // HOST Parquet -> HOST Parquet (cross-host copy)
  if (!registry.has_converter<host_parquet_representation, host_parquet_representation>()) {
    registry.register_converter<host_parquet_representation, host_parquet_representation>(
      detail::convert_host_parquet_to_host_parquet);
  }

  if (!registry
         .has_converter<cached_host_data_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<cached_host_data_representation, cucascade::gpu_table_representation>(
        [&registry](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
          auto r = source.cast<cached_host_data_representation>().get_representation();
          return registry.convert<cucascade::gpu_table_representation>(
            *r, target_memory_space, stream);
        });
  }

  if (!registry.has_converter<cached_host_parquet_representation,
                              cucascade::gpu_table_representation>()) {
    registry
      .register_converter<cached_host_parquet_representation, cucascade::gpu_table_representation>(
        [&registry](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream) {
          auto r = source.cast<cached_host_parquet_representation>().get_representation();
          return registry.convert<cucascade::gpu_table_representation>(
            *r, target_memory_space, stream);
        });
  }
}

}  // namespace sirius
