/*
 * Copyright 2026, Sirius Contributors.
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

#include <string>
#include <string_view>
#include <vector>

namespace sirius::io::s3 {

/// One parsed page of an S3 ListObjectsV2 response.
struct list_objects_v2_page {
  /// Every `<Contents><Key>` in document order, XML-entity-unescaped. Excludes
  /// `<CommonPrefixes><Prefix>` entries (those are directory rollups, not keys).
  std::vector<std::string> keys;
  /// `<IsTruncated>` — true when another page follows.
  bool is_truncated = false;
  /// `<NextContinuationToken>` — the cursor for the next page; empty when absent.
  std::string next_continuation_token;
};

/**
 * @brief Parse one ListObjectsV2 XML response body.
 *
 * Hand-rolled (no XML dependency): the S3 ListObjectsV2 schema is small and
 * fixed. Tolerant of the `xmlns` attribute on `<ListBucketResult>`, leading /
 * trailing whitespace, and a self-closing `<ListBucketResult/>`. XML entities
 * (`&amp; &lt; &gt; &quot; &apos;`) in key / token text are unescaped.
 *
 * @throw std::runtime_error when @p xml is not a recognizable ListObjectsV2
 *        response (no `<ListBucketResult>` — e.g. an S3 `<Error>` body or an
 *        empty string).
 */
list_objects_v2_page parse_list_objects_v2(std::string_view xml);

}  // namespace sirius::io::s3
