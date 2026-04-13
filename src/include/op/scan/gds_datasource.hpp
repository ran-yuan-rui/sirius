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

#include <cudf/io/datasource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <future>
#include <memory>
#include <string>

namespace sirius::op::scan {

/**
 * @brief cudf::io::datasource backed by kvikio::FileHandle for GPU Direct Storage.
 *
 * Supports device_read_async() for DMA from NVMe to GPU HBM via cuFile.
 * Falls back to POSIX reads automatically when GDS hardware or the cuFile
 * driver is absent (KvikIO handles this via compat_mode = AUTO).
 *
 * Uses pimpl to avoid exposing kvikio headers to consumers.
 */
class gds_datasource : public cudf::io::datasource {
 public:
  /**
   * @brief Open a local file with GPU Direct Storage support.
   *
   * @param path  Local file path.
   */
  explicit gds_datasource(std::string const& path);

  ~gds_datasource() override;

  // Move-only (pimpl)
  gds_datasource(gds_datasource&&) noexcept;
  gds_datasource& operator=(gds_datasource&&) noexcept;
  gds_datasource(gds_datasource const&) = delete;
  gds_datasource& operator=(gds_datasource const&) = delete;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override;

  std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  [[nodiscard]] size_t size() const override;

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::scan
