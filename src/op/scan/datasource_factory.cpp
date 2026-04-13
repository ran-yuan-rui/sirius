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

#include <op/scan/datasource_factory.hpp>
#include <op/scan/gds_datasource.hpp>
#include <op/scan/s3_datasource.hpp>

#include <cudf/io/datasource.hpp>

#include <kvikio/shim/cufile.hpp>

#include <stdexcept>
#include <string>

namespace sirius::op::scan {

bool datasource_factory::is_s3_uri(std::string const& uri)
{
  return uri.size() > 5 && (uri.compare(0, 5, "s3://") == 0 || uri.compare(0, 5, "S3://") == 0);
}

bool datasource_factory::is_gds_preferred([[maybe_unused]] std::string const& path)
{
  return kvikio::is_cufile_available();
}

namespace {

std::string strip_file_scheme(std::string const& uri)
{
  // Strip "file://" prefix if present.
  constexpr std::string_view file_prefix = "file://";
  if (uri.size() > file_prefix.size() &&
      uri.compare(0, file_prefix.size(), file_prefix) == 0) {
    return uri.substr(file_prefix.size());
  }
  return uri;
}

}  // namespace

std::unique_ptr<cudf::io::datasource> datasource_factory::create(std::string const& uri,
                                                                  object_store_config const& config)
{
  if (uri.empty()) { throw std::runtime_error("datasource_factory::create: empty URI"); }

  if (is_s3_uri(uri)) {
    return std::make_unique<s3_datasource>(uri, config);
  }

  // Local file path (with optional file:// prefix).
  auto const path = strip_file_scheme(uri);
  if (is_gds_preferred(path)) { return std::make_unique<gds_datasource>(path); }
  return cudf::io::datasource::create(path);
}

std::unique_ptr<cudf::io::datasource> datasource_factory::create(std::string const& path)
{
  if (path.empty()) { throw std::runtime_error("datasource_factory::create: empty path"); }

  if (is_s3_uri(path)) {
    throw std::runtime_error(
      "datasource_factory::create: S3 URI passed to local-path overload. "
      "Use the (uri, config) overload for S3 URIs. URI: " + path);
  }

  auto const local_path = strip_file_scheme(path);
  if (is_gds_preferred(local_path)) { return std::make_unique<gds_datasource>(local_path); }
  return cudf::io::datasource::create(local_path);
}

}  // namespace sirius::op::scan
