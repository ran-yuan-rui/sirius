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

#ifdef SIRIUS_RDMA_SUPPORT

#include "op/scan/object_store_config.hpp"

#include <cudf/io/datasource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <future>
#include <memory>
#include <string>

namespace sirius::op::scan {

/**
 * @brief cudf::io::datasource for S3-compatible stores with RDMA data path.
 *
 * Uses NVIDIA cuObjClient for the RDMA data plane (memory registration, DC transport,
 * RDMA token lifecycle) and libcurl + SigV4 for the HTTP control plane.
 *
 * The control vs. data plane split:
 *   Control: standard S3 GET with `x-amz-rdma-token` header sent via HTTP (libcurl).
 *   Data:    storage server RDMA_WRITEs payload directly into the registered GPU buffer.
 *
 * Requires a cuObjServer-enabled S3 endpoint (e.g., VAST Data, MinIO with RDMA extensions).
 * Standard cloud S3 (AWS, GCS, Azure) is not supported — use s3_datasource instead.
 */
class rdma_s3_datasource : public cudf::io::datasource {
 public:
  /**
   * @brief Construct an RDMA S3 datasource.
   *
   * @param url     S3 object URL (e.g., "s3://bucket/key").
   * @param config  Object store configuration (endpoint, credentials, transport).
   */
  rdma_s3_datasource(std::string const& url, object_store_config const& config);

  ~rdma_s3_datasource() override;

  // Move-only (pimpl)
  rdma_s3_datasource(rdma_s3_datasource&&) noexcept;
  rdma_s3_datasource& operator=(rdma_s3_datasource&&) noexcept;
  rdma_s3_datasource(rdma_s3_datasource const&) = delete;
  rdma_s3_datasource& operator=(rdma_s3_datasource const&) = delete;

  // Host reads for small metadata (footer, PAR1 header) — HTTP Range GET fallback.
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;

  // GPU Direct RDMA reads via cuObjClient.
  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override { return true; }

  std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  [[nodiscard]] size_t size() const override;

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::scan

#endif  // SIRIUS_RDMA_SUPPORT
