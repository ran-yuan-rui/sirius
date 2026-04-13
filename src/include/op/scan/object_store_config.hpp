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

#include <cstdint>
#include <string>

namespace sirius::op::scan {

/**
 * @brief S3 transport mode selection for object store access.
 */
enum class s3_transport {
  AUTO,  ///< Auto-detect: use RDMA if hardware supports it, otherwise HTTP.
  HTTP,  ///< Force HTTP Range GET (works with any S3-compatible endpoint).
  RDMA,  ///< Force RDMA data path (requires cuObjClient + cuObjServer on storage).
};

/**
 * @brief Configuration for S3-compatible object store access.
 *
 * Populated from the `object_store:` section in `sirius.yaml` or via
 * DuckDB `SET` commands (`s3_endpoint`, `s3_region`, etc.).
 *
 * When all fields are left at defaults, KvikIO's RemoteHandle uses
 * `AWS_*` environment variables for credential resolution.
 */
struct object_store_config {
  std::string endpoint;
  std::string region;
  std::string access_key_id;
  std::string secret_access_key;
  std::string session_token;
  s3_transport transport = s3_transport::AUTO;
  bool use_tls = true;
  uint64_t connection_timeout_ms = 30000;
  uint64_t request_timeout_ms = 60000;
};

}  // namespace sirius::op::scan
