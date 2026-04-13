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

#include <op/scan/gds_datasource.hpp>

#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/shim/cufile.hpp>

#include <unistd.h>

#include <cstring>
#include <future>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

namespace {

// Minimum read size to prefer GPU Direct over host-staged reads.
// Small metadata reads (footer, PAR1 header) stay on host via pread.
constexpr size_t GDS_THRESHOLD = 128UL * 1024;  // 128 KB

}  // namespace

// ---------------------------------------------------------------------------
// pimpl
// ---------------------------------------------------------------------------

class gds_datasource::impl {
 public:
  explicit impl(std::string const& path)
    : _handle(path, "r"), _file_size(_handle.nbytes()), _gds_available(kvikio::is_cufile_available())
  {
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst)
  {
    // Use POSIX pread via the file descriptor (no GDS, no O_DIRECT).
    auto bytes = ::pread(_handle.fd(false), dst, size, static_cast<off_t>(offset));
    if (bytes < 0) { throw std::runtime_error("gds_datasource: pread failed"); }
    return static_cast<size_t>(bytes);
  }

  std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* dst)
  {
    // kvikio::FileHandle::pread returns a std::future<size_t>
    return _handle.pread(dst, size, offset);
  }

  [[nodiscard]] bool supports_device_read() const { return _gds_available; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const
  {
    return _gds_available && size >= GDS_THRESHOLD;
  }

  [[nodiscard]] size_t file_size() const { return _file_size; }

 private:
  kvikio::FileHandle _handle;
  size_t _file_size;
  bool _gds_available;
};

// ---------------------------------------------------------------------------
// gds_datasource public API
// ---------------------------------------------------------------------------

gds_datasource::gds_datasource(std::string const& path)
  : _impl(std::make_unique<impl>(path))
{
}

gds_datasource::~gds_datasource() = default;

gds_datasource::gds_datasource(gds_datasource&&) noexcept = default;
gds_datasource& gds_datasource::operator=(gds_datasource&&) noexcept = default;

size_t gds_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return _impl->host_read(offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> gds_datasource::host_read(size_t offset, size_t size)
{
  auto buf = std::make_unique<std::vector<uint8_t>>(size);
  auto bytes_read = _impl->host_read(offset, size, buf->data());

  class owning_buffer : public cudf::io::datasource::buffer {
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

  return std::make_unique<owning_buffer>(std::move(buf), bytes_read);
}

bool gds_datasource::supports_device_read() const { return _impl->supports_device_read(); }

bool gds_datasource::is_device_read_preferred(size_t size) const
{
  return _impl->is_device_read_preferred(size);
}

std::future<size_t> gds_datasource::device_read_async(size_t offset, size_t size, uint8_t* dst,
                                                      [[maybe_unused]] rmm::cuda_stream_view stream)
{
  return _impl->device_read_async(offset, size, dst);
}

size_t gds_datasource::size() const { return _impl->file_size(); }

}  // namespace sirius::op::scan
