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

#include <op/scan/rdma_s3_datasource.hpp>
#include <op/scan/s3_sigv4.hpp>

#include <cuobjclient.h>
#include <curl/curl.h>

#include <cstdlib>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

// ---------------------------------------------------------------------------
// pimpl — wraps cuObjClient + libcurl for the control path
// ---------------------------------------------------------------------------

class rdma_s3_datasource::impl {
 public:
  impl(std::string const& url, object_store_config const& config)
    : _url(url), _config(config), _file_size(0)
  {
    // Initialize cuObjClient with GET/PUT callbacks
    _ops.get = &impl::get_callback;
    _ops.put = nullptr;  // read-only datasource

    _client = std::make_unique<cuObjClient>(_ops, CUOBJ_PROTO_RDMA_DC_V1);
    if (!_client->isConnected()) {
      throw std::runtime_error("rdma_s3_datasource: cuObjClient failed to connect");
    }

    // Determine file size via HTTP HEAD
    _file_size = fetch_size_via_head();
  }

  ~impl() { _client.reset(); }

  size_t host_read(size_t offset, size_t size, uint8_t* dst)
  {
    // For small host reads (metadata), use HTTP Range GET via libcurl directly.
    return http_range_get(offset, size, reinterpret_cast<char*>(dst));
  }

  std::future<size_t> device_read_async(size_t offset, size_t size, uint8_t* d_dst,
                                        rmm::cuda_stream_view stream)
  {
    // Wrap synchronous cuObjGet in std::async for the datasource async interface.
    return std::async(std::launch::async, [this, offset, size, d_dst, stream]() -> size_t {
      return this->device_read(offset, size, d_dst, stream);
    });
  }

  [[nodiscard]] size_t file_size() const { return _file_size; }

 private:
  // Context passed to the cuObjClient GET callback
  struct get_context {
    impl* self;
    size_t offset;
    size_t total_size;
    std::string url;
  };

  size_t device_read(size_t offset, size_t size, uint8_t* d_dst,
                     [[maybe_unused]] rmm::cuda_stream_view stream)
  {
    // Register GPU buffer with cuObjClient
    auto err = _client->cuMemObjGetDescriptor(d_dst, size);
    if (err != CU_OBJ_SUCCESS) {
      throw std::runtime_error("rdma_s3_datasource: cuMemObjGetDescriptor failed");
    }

    // Prepare context for the callback
    get_context ctx{this, offset, size, _url};

    // cuObjGet triggers the GET callback which sends HTTP + RDMA token
    auto bytes = _client->cuObjGet(&ctx, d_dst, size, static_cast<loff_t>(offset));
    if (bytes < 0) {
      _client->cuMemObjPutDescriptor(d_dst);
      throw std::runtime_error("rdma_s3_datasource: cuObjGet failed");
    }

    // Deregister buffer
    _client->cuMemObjPutDescriptor(d_dst);
    return static_cast<size_t>(bytes);
  }

  // cuObjClient GET callback: called with RDMA info, sends HTTP GET with token
  static ssize_t get_callback(const void* handle, char* ptr, size_t size, loff_t offset,
                               const cufileRDMAInfo_t* rdma_info)
  {
    auto* ctx  = static_cast<get_context*>(cuObjClient::getCtx(handle));
    auto* self = ctx->self;

    // Build HTTP GET with Range + x-amz-rdma-token headers
    std::string rdma_token(rdma_info->desc_str, rdma_info->desc_len);

    return self->http_get_with_rdma_token(
      ctx->url, static_cast<size_t>(offset), size, rdma_token);
  }

  ssize_t http_get_with_rdma_token(std::string const& url, size_t offset, size_t size,
                                   std::string const& rdma_token)
  {
    CURL* curl = curl_easy_init();
    if (!curl) { return -1; }

    // Range header
    std::ostringstream range;
    range << offset << "-" << (offset + size - 1);

    // Build headers
    std::unordered_map<std::string, std::string> headers;
    headers["host"]  = extract_host(url);
    headers["range"] = "bytes=" + range.str();

    // SigV4 auth
    auto const auth = s3_sigv4::sign("GET",
                                     url,
                                     headers,
                                     "UNSIGNED-PAYLOAD",
                                     _config.region,
                                     _config.access_key_id,
                                     _config.secret_access_key,
                                     _config.session_token);

    struct curl_slist* header_list = nullptr;
    header_list = curl_slist_append(header_list, ("Authorization: " + auth).c_str());
    header_list = curl_slist_append(header_list, ("x-amz-date: " + s3_sigv4::iso8601_now()).c_str());
    header_list =
      curl_slist_append(header_list, "x-amz-content-sha256: UNSIGNED-PAYLOAD");
    header_list = curl_slist_append(header_list, ("Range: bytes=" + range.str()).c_str());
    header_list =
      curl_slist_append(header_list, ("x-amz-rdma-token: " + rdma_token).c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 0L);

    // Discard response body (data arrives via RDMA, not HTTP body)
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, discard_write_callback);

    auto res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || (http_code != 200 && http_code != 206)) { return -1; }
    return static_cast<ssize_t>(size);
  }

  size_t fetch_size_via_head()
  {
    CURL* curl = curl_easy_init();
    if (!curl) { throw std::runtime_error("rdma_s3_datasource: curl_easy_init failed"); }

    std::unordered_map<std::string, std::string> headers;
    headers["host"] = extract_host(_url);

    auto const auth = s3_sigv4::sign("HEAD",
                                     _url,
                                     headers,
                                     "UNSIGNED-PAYLOAD",
                                     _config.region,
                                     _config.access_key_id,
                                     _config.secret_access_key,
                                     _config.session_token);

    struct curl_slist* header_list = nullptr;
    header_list = curl_slist_append(header_list, ("Authorization: " + auth).c_str());
    header_list = curl_slist_append(header_list, ("x-amz-date: " + s3_sigv4::iso8601_now()).c_str());
    header_list = curl_slist_append(header_list, "x-amz-content-sha256: UNSIGNED-PAYLOAD");

    curl_easy_setopt(curl, CURLOPT_URL, _url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);

    auto res = curl_easy_perform(curl);

    curl_off_t cl = 0;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);

    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || cl <= 0) {
      throw std::runtime_error("rdma_s3_datasource: HEAD request failed for: " + _url);
    }
    return static_cast<size_t>(cl);
  }

  size_t http_range_get(size_t offset, size_t size, char* dst)
  {
    CURL* curl = curl_easy_init();
    if (!curl) { throw std::runtime_error("rdma_s3_datasource: curl_easy_init failed"); }

    std::ostringstream range;
    range << offset << "-" << (offset + size - 1);

    std::unordered_map<std::string, std::string> headers;
    headers["host"]  = extract_host(_url);
    headers["range"] = "bytes=" + range.str();

    auto const auth = s3_sigv4::sign("GET",
                                     _url,
                                     headers,
                                     "UNSIGNED-PAYLOAD",
                                     _config.region,
                                     _config.access_key_id,
                                     _config.secret_access_key,
                                     _config.session_token);

    struct curl_slist* header_list = nullptr;
    header_list = curl_slist_append(header_list, ("Authorization: " + auth).c_str());
    header_list = curl_slist_append(header_list, ("x-amz-date: " + s3_sigv4::iso8601_now()).c_str());
    header_list = curl_slist_append(header_list, "x-amz-content-sha256: UNSIGNED-PAYLOAD");
    header_list = curl_slist_append(header_list, ("Range: bytes=" + range.str()).c_str());

    struct write_state {
      char* buf;
      size_t pos;
      size_t max;
    } ws{dst, 0, size};

    curl_easy_setopt(curl, CURLOPT_URL, _url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, body_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ws);

    auto res = curl_easy_perform(curl);
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
      throw std::runtime_error("rdma_s3_datasource: HTTP Range GET failed");
    }
    return ws.pos;
  }

  static size_t discard_write_callback(char*, size_t size, size_t nmemb, void*) {
    return size * nmemb;
  }

  static size_t body_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata)
  {
    auto* ws     = static_cast<struct write_state*>(userdata);
    auto const n = size * nmemb;
    auto const copy_n = std::min(n, ws->max - ws->pos);
    std::memcpy(ws->buf + ws->pos, ptr, copy_n);
    ws->pos += copy_n;
    return n;
  }

  struct write_state {
    char* buf;
    size_t pos;
    size_t max;
  };

  static std::string extract_host(std::string const& url)
  {
    // Extract host from https://host/path
    auto pos = url.find("://");
    if (pos == std::string::npos) { return ""; }
    auto start = pos + 3;
    auto end   = url.find('/', start);
    return url.substr(start, end - start);
  }

  std::string _url;
  object_store_config _config;
  size_t _file_size;
  CUObjOps_t _ops{};
  std::unique_ptr<cuObjClient> _client;
};

// ---------------------------------------------------------------------------
// rdma_s3_datasource public API
// ---------------------------------------------------------------------------

rdma_s3_datasource::rdma_s3_datasource(std::string const& url, object_store_config const& config)
  : _impl(std::make_unique<impl>(url, config))
{
}

rdma_s3_datasource::~rdma_s3_datasource() = default;

rdma_s3_datasource::rdma_s3_datasource(rdma_s3_datasource&&) noexcept = default;
rdma_s3_datasource& rdma_s3_datasource::operator=(rdma_s3_datasource&&) noexcept = default;

size_t rdma_s3_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return _impl->host_read(offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> rdma_s3_datasource::host_read(size_t offset,
                                                                            size_t size)
{
  auto buf       = std::make_unique<std::vector<uint8_t>>(size);
  auto bytes_read = _impl->host_read(offset, size, buf->data());

  class owning_buffer : public cudf::io::datasource::buffer {
   public:
    owning_buffer(std::unique_ptr<std::vector<uint8_t>> data, size_t len)
      : _data(std::move(data)), _len(len)
    {
    }
    [[nodiscard]] size_t size() const override { return _len; }
    [[nodiscard]] uint8_t const* data() const override { return _data->data(); }

   private:
    std::unique_ptr<std::vector<uint8_t>> _data;
    size_t _len;
  };

  return std::make_unique<owning_buffer>(std::move(buf), bytes_read);
}

std::future<size_t> rdma_s3_datasource::device_read_async(size_t offset, size_t size,
                                                          uint8_t* dst,
                                                          rmm::cuda_stream_view stream)
{
  return _impl->device_read_async(offset, size, dst, stream);
}

size_t rdma_s3_datasource::size() const { return _impl->file_size(); }

}  // namespace sirius::op::scan

#endif  // SIRIUS_RDMA_SUPPORT
