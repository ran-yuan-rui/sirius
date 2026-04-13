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

#include "op/scan/object_store_config.hpp"

#include <cudf/io/datasource.hpp>

#include <cstddef>
#include <future>
#include <memory>
#include <string>

namespace sirius::op::scan {

/**
 * @brief cudf::io::datasource backed by kvikio::RemoteHandle for S3/HTTP object storage.
 *
 * Supports S3, GCS, HTTP/HTTPS, and WebHDFS via KvikIO's remote I/O layer.
 * Credential resolution uses AWS_* environment variables by default, or
 * explicit values from object_store_config.
 *
 * Uses pimpl to avoid exposing kvikio headers to consumers.
 */
class s3_datasource : public cudf::io::datasource {
 public:
  /**
   * @brief Construct an S3 datasource from a URL.
   *
   * @param url     Object URL (e.g., "s3://bucket/key", "https://storage.googleapis.com/...").
   * @param config  Object store configuration for credentials and transport settings.
   */
  explicit s3_datasource(std::string const& url, object_store_config const& config = {});

  ~s3_datasource() override;

  // Move-only (pimpl)
  s3_datasource(s3_datasource&&) noexcept;
  s3_datasource& operator=(s3_datasource&&) noexcept;
  s3_datasource(s3_datasource const&) = delete;
  s3_datasource& operator=(s3_datasource const&) = delete;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] size_t size() const override;

  [[nodiscard]] bool supports_device_read() const override { return false; }

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::scan
