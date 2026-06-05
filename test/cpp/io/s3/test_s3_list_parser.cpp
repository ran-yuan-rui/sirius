/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/s3/s3_list_parser.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using sirius::io::s3::parse_list_objects_v2;

TEST_CASE("parse_list_objects_v2 returns object keys in document order", "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <ListBucketResult>
      <Name>bucket</Name>
      <Prefix>pfx/</Prefix>
      <IsTruncated>false</IsTruncated>
      <Contents><Key>pfx/a.parquet</Key></Contents>
      <Contents><Key>pfx/b.parquet</Key></Contents>
      <Contents><Key>pfx/c.parquet</Key></Contents>
    </ListBucketResult>
  )xml");

  CHECK(page.keys == std::vector<std::string>{"pfx/a.parquet", "pfx/b.parquet", "pfx/c.parquet"});
  CHECK_FALSE(page.is_truncated);
  CHECK(page.next_continuation_token.empty());
}

TEST_CASE("parse_list_objects_v2 preserves pagination state", "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <ListBucketResult>
      <IsTruncated>true</IsTruncated>
      <NextContinuationToken>tok==</NextContinuationToken>
      <Contents><Key>pfx/a.parquet</Key></Contents>
    </ListBucketResult>
  )xml");

  CHECK(page.keys == std::vector<std::string>{"pfx/a.parquet"});
  CHECK(page.is_truncated);
  CHECK(page.next_continuation_token == "tok==");
}

TEST_CASE("parse_list_objects_v2 unescapes XML entities in object keys", "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <ListBucketResult>
      <Contents><Key>a&amp;b/c&lt;d&gt;e&quot;f&apos;g.parquet</Key></Contents>
    </ListBucketResult>
  )xml");

  REQUIRE(page.keys.size() == 1);
  CHECK(page.keys.front() == "a&b/c<d>e\"f'g.parquet");
}

TEST_CASE("parse_list_objects_v2 accepts empty LIST responses", "[s3][list_parser]")
{
  SECTION("self-closing result")
  {
    auto page = parse_list_objects_v2("<ListBucketResult/>");
    CHECK(page.keys.empty());
    CHECK_FALSE(page.is_truncated);
    CHECK(page.next_continuation_token.empty());
  }

  SECTION("no Contents entries")
  {
    auto page = parse_list_objects_v2(R"xml(
      <ListBucketResult>
        <Name>bucket</Name>
        <Prefix>missing/</Prefix>
        <IsTruncated>false</IsTruncated>
      </ListBucketResult>
    )xml");
    CHECK(page.keys.empty());
    CHECK_FALSE(page.is_truncated);
  }
}

TEST_CASE("parse_list_objects_v2 ignores CommonPrefixes entries", "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <ListBucketResult>
      <CommonPrefixes><Prefix>pfx/subdir/</Prefix></CommonPrefixes>
      <Contents><Key>pfx/file.parquet</Key></Contents>
    </ListBucketResult>
  )xml");

  CHECK(page.keys == std::vector<std::string>{"pfx/file.parquet"});
}

TEST_CASE("parse_list_objects_v2 ignores stray Key elements outside Contents", "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <ListBucketResult>
      <Key>not-an-object.parquet</Key>
      <Contents><Key>real.parquet</Key></Contents>
      <Owner><Key>also-not-an-object.parquet</Key></Owner>
    </ListBucketResult>
  )xml");

  CHECK(page.keys == std::vector<std::string>{"real.parquet"});
}

TEST_CASE("parse_list_objects_v2 preserves hive-style key text", "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <ListBucketResult>
      <Contents><Key>t/date=2026-06-01/part-0.parquet</Key></Contents>
      <Contents><Key>t/date=2026-06-02/part-0.parquet</Key></Contents>
    </ListBucketResult>
  )xml");

  CHECK(page.keys == std::vector<std::string>{"t/date=2026-06-01/part-0.parquet",
                                              "t/date=2026-06-02/part-0.parquet"});
}

TEST_CASE("parse_list_objects_v2 tolerates the S3 xmlns attribute and formatting",
          "[s3][list_parser]")
{
  auto page = parse_list_objects_v2(R"xml(
    <?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>sirius-test</Name>
      <Prefix>parquet/</Prefix>
      <IsTruncated>false</IsTruncated>
      <Contents>
        <Key>parquet/nation.parquet</Key>
      </Contents>
    </ListBucketResult>
  )xml");

  CHECK(page.keys == std::vector<std::string>{"parquet/nation.parquet"});
  CHECK_FALSE(page.is_truncated);
}

TEST_CASE("parse_list_objects_v2 rejects non-LIST response bodies", "[s3][list_parser]")
{
  CHECK_THROWS_AS(parse_list_objects_v2(""), std::runtime_error);
  CHECK_THROWS_AS(parse_list_objects_v2(R"xml(
    <Error>
      <Code>NoSuchBucket</Code>
      <Message>The specified bucket does not exist</Message>
    </Error>
  )xml"),
                  std::runtime_error);
}
