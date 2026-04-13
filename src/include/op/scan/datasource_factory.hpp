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

#include <memory>
#include <string>

namespace sirius::op::scan {

/**
 * @brief Factory that creates the appropriate cudf::io::datasource for a given URI.
 *
 * Dispatch logic:
 *   "/path" or "file:///path"  ->  cudf::io::datasource (or gds_datasource if NVMe + GDS)
 *   "s3://bucket/key"          ->  config.transport == RDMA ? rdma_s3_datasource
 *                                                           : s3_datasource
 *
 * Replaces the 3 hardcoded cudf::io::datasource::create(path) call sites in the scan pipeline.
 */
class datasource_factory {
 public:
  /**
   * @brief Create a datasource from a URI with object store configuration.
   *
   * @param uri     File path or URI (e.g., "/data/file.parquet", "s3://bucket/key").
   * @param config  Object store configuration (credentials, transport, endpoints).
   * @return A cudf::io::datasource instance for the given URI.
   * @throws std::runtime_error if the URI scheme is unsupported or the backend is unavailable.
   */
  static std::unique_ptr<cudf::io::datasource> create(std::string const& uri,
                                                       object_store_config const& config);

  /**
   * @brief Convenience overload for local file paths (no object store config needed).
   *
   * Equivalent to cudf::io::datasource::create(path) today, but will route through
   * GDS when available.
   *
   * @param path  Local file path.
   * @return A cudf::io::datasource instance.
   */
  static std::unique_ptr<cudf::io::datasource> create(std::string const& path);

 private:
  static bool is_s3_uri(std::string const& uri);
  static bool is_gds_preferred(std::string const& path);
};

}  // namespace sirius::op::scan
