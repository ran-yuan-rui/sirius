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

#ifdef SIRIUS_RDMA_SUPPORT

#include <string>
#include <unordered_map>

namespace sirius::op::scan {

/**
 * @brief AWS Signature Version 4 signing for S3 HTTP requests.
 *
 * Self-contained SigV4 implementation using OpenSSL HMAC-SHA256.
 * Computes the Authorization header value for S3 API requests.
 */
struct s3_sigv4 {
  /**
   * @brief Sign an S3 HTTP request using AWS SigV4.
   *
   * @param method      HTTP method (e.g., "GET", "HEAD").
   * @param url         Full URL (e.g., "https://bucket.s3.us-east-1.amazonaws.com/key").
   * @param headers     Request headers (key -> value). Must include "host".
   * @param payload_hash SHA256 hex digest of the request body (or "UNSIGNED-PAYLOAD").
   * @param region      AWS region (e.g., "us-east-1").
   * @param access_key  AWS access key ID.
   * @param secret_key  AWS secret access key.
   * @param session_token Optional session token (empty if not using temporary credentials).
   * @return Authorization header value string.
   */
  static std::string sign(std::string const& method,
                          std::string const& url,
                          std::unordered_map<std::string, std::string> const& headers,
                          std::string const& payload_hash,
                          std::string const& region,
                          std::string const& access_key,
                          std::string const& secret_key,
                          std::string const& session_token = "");

  /// Compute SHA256 hex digest of data.
  static std::string sha256_hex(std::string const& data);

  /// Compute HMAC-SHA256 of data with key (returns raw bytes).
  static std::string hmac_sha256(std::string const& key, std::string const& data);

  /// Get current UTC time in ISO 8601 format (YYYYMMDDTHHMMSSZ).
  static std::string iso8601_now();

  /// Get current UTC date (YYYYMMDD).
  static std::string date_now();
};

}  // namespace sirius::op::scan

#endif  // SIRIUS_RDMA_SUPPORT
