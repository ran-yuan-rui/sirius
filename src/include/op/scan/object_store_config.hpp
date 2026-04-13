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
#include <string_view>
#include <unordered_map>

namespace sirius::op::scan {

/**
 * @brief S3 transport mode selection for object store access.
 */
enum class s3_transport {
  AUTO,  ///< Auto-detect: use RDMA if hardware supports it, otherwise HTTP.
  HTTP,  ///< Force HTTP Range GET (works with any S3-compatible endpoint).
  RDMA,  ///< Force RDMA data path (requires cuObjClient + cuObjServer on storage).
};

inline bool string_to_enum(std::string_view sv, s3_transport& t)
{
  static const std::unordered_map<std::string_view, s3_transport> map = {
    {"auto", s3_transport::AUTO},
    {"http", s3_transport::HTTP},
    {"https", s3_transport::HTTP},
    {"rdma", s3_transport::RDMA},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    t = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(s3_transport t, std::string& s)
{
  switch (t) {
    case s3_transport::AUTO: s = "auto"; return true;
    case s3_transport::HTTP: s = "http"; return true;
    case s3_transport::RDMA: s = "rdma"; return true;
  }
  return false;
}

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
