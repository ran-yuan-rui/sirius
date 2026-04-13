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
#include "yaml_reader.hpp"

#include <op/scan/object_store_config.hpp>

#include <yaml-cpp/yaml.h>

using namespace sirius::op::scan;

TEST_CASE("object_store_config defaults", "[config_opt][datasource]")
{
  object_store_config cfg;
  REQUIRE(cfg.endpoint.empty());
  REQUIRE(cfg.region.empty());
  REQUIRE(cfg.access_key_id.empty());
  REQUIRE(cfg.secret_access_key.empty());
  REQUIRE(cfg.session_token.empty());
  REQUIRE(cfg.transport == s3_transport::AUTO);
  REQUIRE(cfg.use_tls == true);
  REQUIRE(cfg.connection_timeout_ms == 30000);
  REQUIRE(cfg.request_timeout_ms == 60000);
}

TEST_CASE("object_store_config transport enum string_to_enum", "[config_opt][datasource]")
{
  s3_transport t;

  REQUIRE(string_to_enum("auto", t));
  REQUIRE(t == s3_transport::AUTO);

  REQUIRE(string_to_enum("http", t));
  REQUIRE(t == s3_transport::HTTP);

  REQUIRE(string_to_enum("https", t));
  REQUIRE(t == s3_transport::HTTP);

  REQUIRE(string_to_enum("rdma", t));
  REQUIRE(t == s3_transport::RDMA);

  REQUIRE_FALSE(string_to_enum("invalid", t));
}

TEST_CASE("object_store_config transport enum_to_string", "[config_opt][datasource]")
{
  std::string s;

  REQUIRE(enum_to_string(s3_transport::AUTO, s));
  REQUIRE(s == "auto");

  REQUIRE(enum_to_string(s3_transport::HTTP, s));
  REQUIRE(s == "http");

  REQUIRE(enum_to_string(s3_transport::RDMA, s));
  REQUIRE(s == "rdma");
}

TEST_CASE("object_store_config yaml parsing", "[config_opt][datasource]")
{
  auto node = YAML::Load(R"(
    endpoint: "https://s3.us-west-2.amazonaws.com"
    region: "us-west-2"
    access_key_id: "AKIAIOSFODNN7EXAMPLE"
    secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    session_token: "FwoGZXIvYXdzEBYaDH..."
    transport: "rdma"
    use_tls: false
    connection_timeout_ms: 5000
    request_timeout_ms: 120000
  )");

  object_store_config cfg;
  sirius::yaml::reader r(node, "object_store");
  r.optional("endpoint", cfg.endpoint);
  r.optional("region", cfg.region);
  r.optional("access_key_id", cfg.access_key_id);
  r.optional("secret_access_key", cfg.secret_access_key);
  r.optional("session_token", cfg.session_token);
  r.optional("transport", cfg.transport);
  r.optional("use_tls", cfg.use_tls);
  r.optional("connection_timeout_ms", cfg.connection_timeout_ms);
  r.optional("request_timeout_ms", cfg.request_timeout_ms);
  r.reject_unknown();

  REQUIRE(cfg.endpoint == "https://s3.us-west-2.amazonaws.com");
  REQUIRE(cfg.region == "us-west-2");
  REQUIRE(cfg.access_key_id == "AKIAIOSFODNN7EXAMPLE");
  REQUIRE(cfg.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  REQUIRE(cfg.session_token == "FwoGZXIvYXdzEBYaDH...");
  REQUIRE(cfg.transport == s3_transport::RDMA);
  REQUIRE(cfg.use_tls == false);
  REQUIRE(cfg.connection_timeout_ms == 5000);
  REQUIRE(cfg.request_timeout_ms == 120000);
}

TEST_CASE("object_store_config yaml defaults when empty", "[config_opt][datasource]")
{
  auto node = YAML::Load("{}");
  object_store_config cfg;
  sirius::yaml::reader r(node, "object_store");
  r.optional("endpoint", cfg.endpoint);
  r.optional("transport", cfg.transport);
  r.reject_unknown();

  REQUIRE(cfg.endpoint.empty());
  REQUIRE(cfg.transport == s3_transport::AUTO);
}
