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

#include "io/s3/s3_ioctx.hpp"

#include "io/s3/s3_list_parser.hpp"
#include "io/s3/sigv4.hpp"
#include "io/uri_parser.hpp"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io::s3 {

s3_ioctx::s3_ioctx(std::shared_ptr<s3_request_authorizer> creds,
                   long request_timeout_s,
                   std::string ca_bundle_path,
                   bool tls_verify,
                   std::size_t max_connections,
                   cucascade::memory::fixed_size_host_memory_resource* host_mr,
                   std::optional<std::size_t> max_retry_attempts,
                   std::optional<std::chrono::milliseconds> retry_backoff_base,
                   std::optional<std::chrono::milliseconds> retry_jitter,
                   std::optional<bool> honor_retry_after)
  : templated_ioctx<s3_reactor>(1,
                                [creds = std::move(creds),
                                 request_timeout_s,
                                 ca_bundle_path = std::move(ca_bundle_path),
                                 tls_verify,
                                 max_connections,
                                 host_mr,
                                 max_retry_attempts,
                                 retry_backoff_base,
                                 retry_jitter,
                                 honor_retry_after]() {
                                  s3_reactor::config cfg;
                                  cfg.creds                = creds;
                                  cfg.request_timeout_s    = request_timeout_s;
                                  cfg.ca_bundle_path       = ca_bundle_path;
                                  cfg.tls_verify           = tls_verify;
                                  cfg.max_connections      = max_connections;
                                  cfg.host_memory_resource = host_mr;
                                  // Only override the reactor's own retry defaults
                                  // when explicitly provided (no duplicated literals).
                                  if (max_retry_attempts)
                                    cfg.max_retry_attempts = *max_retry_attempts;
                                  if (retry_backoff_base)
                                    cfg.retry_backoff_base = *retry_backoff_base;
                                  if (retry_jitter) cfg.retry_jitter = *retry_jitter;
                                  if (honor_retry_after) cfg.honor_retry_after = *honor_retry_after;
                                  return std::make_unique<s3_reactor>(std::move(cfg));
                                })
{
}

std::shared_ptr<sirius_io_object> s3_ioctx::create_io_object(std::string path)
{
  auto parsed = sirius::io::parse(path);
  if (parsed.scheme != "s3") {
    throw std::invalid_argument("s3_ioctx::create_io_object: unsupported scheme '" + parsed.scheme +
                                "'");
  }
  auto size  = reactor().head_object_size(parsed.host, parsed.path);
  auto state = std::make_shared<s3_object_state>(
    s3_object_state{std::move(parsed.host), std::move(parsed.path), size});
  return std::make_shared<s3_async_io_object>(std::move(path), std::move(state));
}

void s3_ioctx::host_read_ranges_async_io(sirius_io_object& obj,
                                         std::vector<cudf::io::text::byte_range_info> const& ranges,
                                         std::span<cudf::host_span<std::byte>> dst,
                                         io_completion_handler handler)
{
  // Hard contract: never sync-throw; all errors via the handler's exception_ptr;
  // handler fires exactly once.
  if (ranges.size() != dst.size()) {
    handler(0,
            std::make_exception_ptr(
              std::invalid_argument("s3_ioctx::host_read_ranges: ranges/dst size mismatch")));
    return;
  }
  if (ranges.empty()) {
    handler(0, nullptr);  // zero-work success may complete inline
    return;
  }

  auto* tobj = dynamic_cast<s3_async_io_object*>(&obj);
  if (tobj == nullptr) {
    handler(0,
            std::make_exception_ptr(
              std::invalid_argument("s3_ioctx::host_read_ranges: io_object is not an "
                                    "s3_async_io_object")));
    return;
  }
  auto const file_size = tobj->size();

  std::vector<s3_reactor::host_read_req_type> reqs;
  reqs.reserve(ranges.size());
  std::size_t total = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto const off    = static_cast<std::size_t>(ranges[i].offset());
    auto const wanted = static_cast<std::size_t>(ranges[i].size());
    auto const sz     = std::min(wanted, file_size > off ? file_size - off : std::size_t{0});
    if (dst[i].size() < sz) {
      handler(0,
              std::make_exception_ptr(
                std::invalid_argument("s3_ioctx::host_read_ranges: dst span too small")));
      return;
    }
    if (sz == 0) continue;
    s3_reactor::host_read_req_type req;
    req.handle = tobj->host_handle();
    req.offset = off;
    req.size   = sz;
    req.dst    = reinterpret_cast<std::uint8_t*>(dst[i].data());
    reqs.push_back(std::move(req));
    total += sz;
  }

  auto ctx = request_context::create(reqs.size(), total, std::move(handler));
  if (!ctx) return;  // zero valid reqs -> create() already fired the handler
  for (auto& r : reqs)
    r.ctx = ctx;
  reactor().host_enqueue_bulk(std::span<s3_reactor::host_read_req_type>(reqs.data(), reqs.size()));
}

std::uint64_t s3_ioctx::bytes_read_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->bytes_read_total();
  return total;
}

std::uint64_t s3_ioctx::fsmr_borrows_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->fsmr_borrows_total();
  return total;
}

std::uint64_t s3_ioctx::device_copies_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->device_copies_total();
  return total;
}

std::uint64_t s3_ioctx::device_stream_sync_total() const noexcept
{
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->device_stream_sync_total();
  return total;
}

std::uint64_t s3_ioctx::device_peak_inflight() const noexcept
{
  // sum the per-reactor peaks: a safe upper bound on concurrent staging blocks
  // (exact for the single-reactor S3 backend).
  std::uint64_t total = 0;
  for (auto const& r : _reactors)
    total += r->device_peak_inflight();
  return total;
}

std::size_t s3_ioctx::head_object_size(std::string_view bucket, std::string_view key)
{
  return reactor().head_object_size(bucket, key);
}

std::vector<std::string> s3_ioctx::list_objects(std::string_view bucket,
                                                std::string_view prefix,
                                                unsigned page_size,
                                                std::size_t max_keys)
{
  // ListObjectsV2 caps max-keys at 1000; clamp (0 -> default) so a caller can't
  // ask for an out-of-range page.
  unsigned const per_page = (page_size == 0 || page_size > 1000) ? 1000U : page_size;

  std::vector<std::string> keys;
  std::string token;
  do {
    // Canonical query with components in sorted (alphabetical) key order, as
    // SigV4 requires: continuation-token < list-type < max-keys < prefix.
    std::string query;
    if (!token.empty()) {
      query += "continuation-token=";
      query += uri_encode(token, /*encode_slash=*/true);
      query += '&';
    }
    query += "list-type=2&max-keys=";
    query += std::to_string(per_page);
    query += "&prefix=";
    query += uri_encode(prefix, /*encode_slash=*/true);

    auto const body = reactor().blocking_list_request(bucket, query);
    auto page       = parse_list_objects_v2(body);
    keys.insert(keys.end(),
                std::make_move_iterator(page.keys.begin()),
                std::make_move_iterator(page.keys.end()));
    token = page.is_truncated ? std::move(page.next_continuation_token) : std::string{};

    // A truncated page must carry a continuation token. A non-compliant response
    // (IsTruncated=true with no NextContinuationToken) would otherwise exit the
    // loop here and silently return only the pages seen so far — a partial key
    // set. Refuse it rather than under-report the listing.
    if (page.is_truncated && token.empty()) {
      throw std::runtime_error(
        "s3_ioctx::list_objects: truncated listing with no continuation token for prefix '" +
        std::string{prefix} + "' in bucket '" + std::string{bucket} +
        "'; refusing to return a partial result");
    }

    // Memory guard: throw rather than truncate. A silently-truncated key set
    // would later resolve a glob to a partial table (a wrong result), so this
    // must be a hard error the caller surfaces, not a logged warning.
    if (keys.size() > max_keys) {
      throw std::runtime_error("s3_ioctx::list_objects: S3 LIST matched more than " +
                               std::to_string(max_keys) + " objects under prefix '" +
                               std::string{prefix} + "'; narrow the prefix or raise max_keys");
    }
  } while (!token.empty());

  return keys;
}

}  // namespace sirius::io::s3
