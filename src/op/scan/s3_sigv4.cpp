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

#ifdef SIRIUS_RDMA_SUPPORT

#include <op/scan/s3_sigv4.hpp>

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace sirius::op::scan {

std::string s3_sigv4::sha256_hex(std::string const& data)
{
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int len = 0;
  auto* ctx        = EVP_MD_CTX_new();
  EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
  EVP_DigestUpdate(ctx, data.data(), data.size());
  EVP_DigestFinal_ex(ctx, hash, &len);
  EVP_MD_CTX_free(ctx);

  std::ostringstream ss;
  for (unsigned int i = 0; i < len; ++i) {
    ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
  }
  return ss.str();
}

std::string s3_sigv4::hmac_sha256(std::string const& key, std::string const& data)
{
  unsigned char result[EVP_MAX_MD_SIZE];
  unsigned int len = 0;
  HMAC(EVP_sha256(),
       key.data(),
       static_cast<int>(key.size()),
       reinterpret_cast<unsigned char const*>(data.data()),
       data.size(),
       result,
       &len);
  return {reinterpret_cast<char const*>(result), len};
}

std::string s3_sigv4::iso8601_now()
{
  auto t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

std::string s3_sigv4::date_now()
{
  auto t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[16];
  std::strftime(buf, sizeof(buf), "%Y%m%d", &tm);
  return buf;
}

namespace {

std::string uri_encode(std::string const& s, bool encode_slash = true)
{
  std::ostringstream encoded;
  for (unsigned char c : s) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
        c == '_' || c == '.' || c == '~' || (!encode_slash && c == '/')) {
      encoded << c;
    } else {
      encoded << '%' << std::uppercase << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<int>(c);
    }
  }
  return encoded.str();
}

// Extract host and path from URL
struct url_parts {
  std::string host;
  std::string path;
  std::string query;
};

url_parts parse_url(std::string const& url)
{
  // Match: https?://host/path?query
  std::regex re(R"(https?://([^/?#]+)(/[^?#]*)?\??(.*)?)", std::regex_constants::icase);
  std::smatch m;
  if (!std::regex_match(url, m, re)) { throw std::runtime_error("s3_sigv4: invalid URL: " + url); }
  return {m[1].str(), m[2].matched ? m[2].str() : "/", m[3].matched ? m[3].str() : ""};
}

}  // namespace

std::string s3_sigv4::sign(std::string const& method,
                           std::string const& url,
                           std::unordered_map<std::string, std::string> const& headers,
                           std::string const& payload_hash,
                           std::string const& region,
                           std::string const& access_key,
                           std::string const& secret_key,
                           std::string const& session_token)
{
  auto const amz_date = iso8601_now();
  auto const date     = amz_date.substr(0, 8);
  auto const parts    = parse_url(url);

  // Canonical headers: lowercase, sorted by key, trimmed values
  std::map<std::string, std::string> canonical_headers;
  for (auto const& [k, v] : headers) {
    std::string lower_key = k;
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
    canonical_headers[lower_key] = v;
  }
  canonical_headers["x-amz-date"]          = amz_date;
  canonical_headers["x-amz-content-sha256"] = payload_hash;
  if (!session_token.empty()) { canonical_headers["x-amz-security-token"] = session_token; }

  // Build canonical headers string and signed headers list
  std::ostringstream canonical_headers_str;
  std::ostringstream signed_headers_str;
  bool first = true;
  for (auto const& [k, v] : canonical_headers) {
    canonical_headers_str << k << ":" << v << "\n";
    if (!first) { signed_headers_str << ";"; }
    signed_headers_str << k;
    first = false;
  }
  auto const signed_headers = signed_headers_str.str();

  // Canonical request
  auto const canonical_uri = uri_encode(parts.path, false);
  std::ostringstream canonical_request;
  canonical_request << method << "\n"
                    << canonical_uri << "\n"
                    << parts.query << "\n"
                    << canonical_headers_str.str() << "\n"
                    << signed_headers << "\n"
                    << payload_hash;

  // String to sign
  auto const credential_scope = date + "/" + region + "/s3/aws4_request";
  std::ostringstream string_to_sign;
  string_to_sign << "AWS4-HMAC-SHA256\n"
                 << amz_date << "\n"
                 << credential_scope << "\n"
                 << sha256_hex(canonical_request.str());

  // Signing key
  auto const k_date    = hmac_sha256("AWS4" + secret_key, date);
  auto const k_region  = hmac_sha256(k_date, region);
  auto const k_service = hmac_sha256(k_region, "s3");
  auto const k_signing = hmac_sha256(k_service, "aws4_request");

  // Signature
  auto const signature_raw = hmac_sha256(k_signing, string_to_sign.str());
  std::ostringstream signature_hex;
  for (unsigned char c : signature_raw) {
    signature_hex << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
  }

  return "AWS4-HMAC-SHA256 Credential=" + access_key + "/" + credential_scope +
         ", SignedHeaders=" + signed_headers + ", Signature=" + signature_hex.str();
}

}  // namespace sirius::op::scan

#endif  // SIRIUS_RDMA_SUPPORT
