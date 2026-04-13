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

#include "catch.hpp"

#include <op/scan/gds_datasource.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

using namespace sirius::op::scan;

namespace {

std::string write_test_file(std::string const& dir, std::string const& content)
{
  auto path = dir + "/test_gds.bin";
  std::ofstream f(path, std::ios::binary);
  f.write(content.data(), static_cast<std::streamsize>(content.size()));
  f.close();
  return path;
}

}  // namespace

TEST_CASE("gds_datasource size matches file size", "[datasource][gds]")
{
  std::string content = "Hello GDS datasource test data!";
  auto path           = write_test_file("/tmp", content);
  gds_datasource ds(path);
  REQUIRE(ds.size() == content.size());
  std::remove(path.c_str());
}

TEST_CASE("gds_datasource host_read matches file content", "[datasource][gds]")
{
  std::string content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  auto path           = write_test_file("/tmp", content);
  gds_datasource ds(path);

  // Full read
  std::vector<uint8_t> buf(content.size());
  auto bytes = ds.host_read(0, content.size(), buf.data());
  REQUIRE(bytes == content.size());
  REQUIRE(std::memcmp(buf.data(), content.data(), content.size()) == 0);

  // Partial read at offset
  std::vector<uint8_t> partial(10);
  bytes = ds.host_read(5, 10, partial.data());
  REQUIRE(bytes == 10);
  REQUIRE(std::memcmp(partial.data(), content.data() + 5, 10) == 0);

  std::remove(path.c_str());
}

TEST_CASE("gds_datasource host_read buffer overload", "[datasource][gds]")
{
  std::string content = "Test buffer overload";
  auto path           = write_test_file("/tmp", content);
  gds_datasource ds(path);

  auto buf = ds.host_read(0, content.size());
  REQUIRE(buf != nullptr);
  REQUIRE(buf->size() == content.size());
  REQUIRE(std::memcmp(buf->data(), content.data(), content.size()) == 0);

  std::remove(path.c_str());
}

TEST_CASE("gds_datasource supports_device_read reflects hardware", "[datasource][gds]")
{
  std::string content = "x";
  auto path           = write_test_file("/tmp", content);
  gds_datasource ds(path);

  // On systems without GDS: supports_device_read() == false
  // On systems with GDS: supports_device_read() == true
  // Either way, it should not crash.
  [[maybe_unused]] auto result = ds.supports_device_read();
  std::remove(path.c_str());
}
