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

#include <op/scan/s3_datasource.hpp>

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include <future>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

// ---------------------------------------------------------------------------
// pimpl
// ---------------------------------------------------------------------------

class s3_datasource::impl {
 public:
  impl(std::string const& url, [[maybe_unused]] object_store_config const& config)
    : _handle(url), _file_size(_handle.nbytes())
  {
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst)
  {
    return _handle.pread(dst, size, offset).get();
  }

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst)
  {
    return _handle.pread(dst, size, offset);
  }

  [[nodiscard]] size_t file_size() const { return _file_size; }

 private:
  kvikio::RemoteHandle _handle;
  size_t _file_size;
};

// ---------------------------------------------------------------------------
// s3_datasource public API
// ---------------------------------------------------------------------------

s3_datasource::s3_datasource(std::string const& url, object_store_config const& config)
  : _impl(std::make_unique<impl>(url, config))
{
}

s3_datasource::~s3_datasource() = default;

s3_datasource::s3_datasource(s3_datasource&&) noexcept = default;
s3_datasource& s3_datasource::operator=(s3_datasource&&) noexcept = default;

size_t s3_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return _impl->host_read(offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> s3_datasource::host_read(size_t offset, size_t size)
{
  auto buf = std::make_unique<std::vector<uint8_t>>(size);
  auto bytes_read = _impl->host_read(offset, size, buf->data());

  // cudf::io::datasource::buffer is an interface; create a concrete one by wrapping the vector.
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

std::future<size_t> s3_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return _impl->host_read_async(offset, size, dst);
}

size_t s3_datasource::size() const { return _impl->file_size(); }

}  // namespace sirius::op::scan
