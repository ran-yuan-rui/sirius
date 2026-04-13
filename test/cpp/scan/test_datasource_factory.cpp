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

#include <op/scan/datasource_factory.hpp>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace sirius::op::scan;

namespace {

// Write a minimal valid Parquet file (PAR1 magic + empty footer + footer length + PAR1 magic).
// This is enough for datasource::size() to return a non-zero value.
std::string write_tiny_parquet(std::string const& dir)
{
  auto path = dir + "/test_factory.parquet";
  std::ofstream f(path, std::ios::binary);
  // PAR1 magic
  f.write("PAR1", 4);
  // Minimal Thrift-encoded FileMetaData: version=1, num_rows=0, no row groups, no schema
  // (15 bytes of valid Thrift compact protocol)
  unsigned char footer[] = {0x15, 0x02, 0x15, 0x00, 0x15, 0x00, 0x15, 0x00, 0x15, 0x00, 0x00};
  f.write(reinterpret_cast<char const*>(footer), sizeof(footer));
  // Footer length (4 bytes, little-endian)
  uint32_t footer_len = sizeof(footer);
  f.write(reinterpret_cast<char const*>(&footer_len), 4);
  // PAR1 magic
  f.write("PAR1", 4);
  f.close();
  return path;
}

}  // namespace

TEST_CASE("datasource_factory local file", "[datasource]")
{
  auto path = write_tiny_parquet("/tmp");
  auto ds   = datasource_factory::create(path);
  REQUIRE(ds != nullptr);
  REQUIRE(ds->size() > 0);
  std::remove(path.c_str());
}

TEST_CASE("datasource_factory file:// scheme", "[datasource]")
{
  auto path = write_tiny_parquet("/tmp");
  auto ds   = datasource_factory::create("file://" + path);
  REQUIRE(ds != nullptr);
  REQUIRE(ds->size() > 0);
  std::remove(path.c_str());
}

TEST_CASE("datasource_factory throws on s3:// uri", "[datasource]")
{
  object_store_config config;
  REQUIRE_THROWS_AS(datasource_factory::create("s3://bucket/key", config), std::runtime_error);
}

TEST_CASE("datasource_factory throws on s3:// uri without config", "[datasource]")
{
  REQUIRE_THROWS_AS(datasource_factory::create("s3://bucket/key"), std::runtime_error);
}

TEST_CASE("datasource_factory empty path throws", "[datasource]")
{
  REQUIRE_THROWS_AS(datasource_factory::create(""), std::runtime_error);
}

TEST_CASE("datasource_factory local file with config overload", "[datasource]")
{
  auto path = write_tiny_parquet("/tmp");
  object_store_config config;
  auto ds = datasource_factory::create(path, config);
  REQUIRE(ds != nullptr);
  REQUIRE(ds->size() > 0);
  std::remove(path.c_str());
}
