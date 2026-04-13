# Implementation Plan: Multi-Backend Datasource Layer

**Based on:** [sirius-multidatasource-proposal.md](sirius-multidatasource-proposal.md) and [gap.md](gap.md)
**Date:** 2026-04-13

Each PR compiles, passes existing regression tests, and introduces no new failures.

---

## Conventions

- **Tests:** Catch2, tags `[datasource]` for isolated tests, `[shared_context][datasource]` for scan-integrated tests.
- **Headers:** `src/include/op/scan/` for scan-layer interfaces, `src/include/data/` for data representations.
- **Sources:** `src/op/scan/` for implementations. Add to `EXTENSION_SOURCES` in `CMakeLists.txt`.
- **Build check:** `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` after each PR.
- **Regression:** `build/release/extension/sirius/test/cpp/sirius_unittest` after each PR.
- **Code quality:** `pre-commit run -a` (or `pixi run pre-commit run -a`) before each commit. Hooks enforce clang-format, codespell, trailing whitespace, end-of-file fixer. If hooks modify files, stage and re-commit.
- **Commit message:** Include `Co-Authored-By` trailer for AI-assisted commits per project convention.

## AI-Assisted Development

Before implementing each PR, use the project's Claude Code skills:

| When | Skill | Why |
|------|-------|-----|
| Before writing any new operator, datasource, or pipeline code | `/module-context <task>` | Loads cudf, rmm, duckdb, cucascade, kvikio API docs so you have accurate function signatures and usage patterns |
| When the build fails | `/build-errors` | Analyzes errors, suggests fixes, iteratively rebuilds |
| After implementation, to run tests | `/test` | Builds and runs the test suite interactively |
| When adding new dependency modules (cuObjClient, libcurl) | `/module-discover <library>` | Generates LLM-consumable API docs for the new dependency |

---

## PR 1: Interface Definitions (Review Scope) -- DONE

**Status:** Committed as `2951d4a`

### New header files

All headers contain only class/struct declarations, enums, and `#include` guards. No `.cpp` files, no `CMakeLists.txt` changes. The headers are not yet `#include`d by any existing code, so the build is unaffected.

**`src/include/op/scan/object_store_config.hpp`**

```cpp
namespace sirius::op::scan {

enum class s3_transport { AUTO, HTTP, RDMA };

struct object_store_config {
  std::string endpoint;
  std::string region;
  std::string access_key_id;
  std::string secret_access_key;
  std::string session_token;
  s3_transport transport = s3_transport::AUTO;
  bool use_tls = true;
  uint64_t connection_timeout_ms = 30000;
  uint64_t request_timeout_ms = 60000;
};

}  // namespace sirius::op::scan
```

**`src/include/op/scan/datasource_factory.hpp`**

```cpp
namespace sirius::op::scan {

class datasource_factory {
 public:
  static std::unique_ptr<cudf::io::datasource> create(
    std::string const& uri,
    object_store_config const& config);

  static std::unique_ptr<cudf::io::datasource> create(std::string const& path);

 private:
  static bool is_s3_uri(std::string const& uri);
  static bool is_gds_preferred(std::string const& path);
};

}  // namespace sirius::op::scan
```

**`src/include/op/scan/s3_datasource.hpp`**

```cpp
namespace sirius::op::scan {

/// cudf::io::datasource backed by kvikio::RemoteHandle for S3/HTTP object storage.
class s3_datasource : public cudf::io::datasource {
 public:
  explicit s3_datasource(std::string const& url,
                         object_store_config const& config = {});

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;
  size_t size() const override;
  bool supports_device_read() const override { return false; }

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::scan
```

Use pimpl to avoid exposing `kvikio::RemoteHandle` in the header (KvikIO headers may not be available until the build system change).

**`src/include/op/scan/gds_datasource.hpp`**

```cpp
namespace sirius::op::scan {

/// cudf::io::datasource backed by kvikio::FileHandle for GPU Direct Storage.
class gds_datasource : public cudf::io::datasource {
 public:
  explicit gds_datasource(std::string const& path);
  ~gds_datasource();

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  bool supports_device_read() const override;
  bool is_device_read_preferred(size_t size) const override;
  std::future<size_t> device_read_async(
    size_t offset, size_t size, uint8_t* dst,
    rmm::cuda_stream_view stream) override;
  size_t size() const override;

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::scan
```

**`src/include/op/scan/rdma_s3_datasource.hpp`**

Guarded by `#ifdef SIRIUS_RDMA_SUPPORT`.

```cpp
namespace sirius::op::scan {

/// cudf::io::datasource for S3-compatible stores with RDMA data path.
/// HTTP/HTTPS control plane (S3 GET + RDMA token header).
/// Data arrives via RDMA_WRITE directly into GPU HBM.
class rdma_s3_datasource : public cudf::io::datasource {
 public:
  rdma_s3_datasource(std::string const& url, object_store_config const& config);
  ~rdma_s3_datasource();

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  bool supports_device_read() const override { return true; }
  bool is_device_read_preferred(size_t size) const override { return true; }
  std::future<size_t> device_read_async(
    size_t offset, size_t size, uint8_t* dst,
    rmm::cuda_stream_view stream) override;
  size_t size() const override;

 private:
  struct rdma_context;
  std::unique_ptr<rdma_context> _ctx;
  std::string _url;
  object_store_config _config;
  size_t _file_size;
};

}  // namespace sirius::op::scan
```

**`src/include/data/gpu_parquet_representation.hpp`**

```cpp
namespace sirius::data {

/// Compressed Parquet column chunks resident in GPU memory.
/// Used by the GPU Direct path to avoid host-memory staging.
class gpu_parquet_representation : public cucascade::idata_representation {
 public:
  gpu_parquet_representation(
    cucascade::memory::memory_space& memory_space,
    rmm::device_buffer column_chunks,
    std::shared_ptr<hybrid_scan_reader> parquet_reader,
    cudf::io::parquet_reader_options reader_options,
    std::vector<cudf::size_type> row_group_indices,
    std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges,
    std::size_t size_in_bytes,
    std::size_t uncompressed_size_in_bytes,
    std::size_t file_size,
    std::shared_ptr<cudf::io::datasource> fallback_datasource,
    std::shared_ptr<translated_expression> filter_expression = nullptr,
    std::vector<std::size_t> post_filter_projection_ids = {});

  [[nodiscard]] std::size_t get_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;
  [[nodiscard]] std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override;

  [[nodiscard]] rmm::device_buffer const& get_column_chunks() const;
  [[nodiscard]] cudf::io::parquet_reader_options const& get_reader_options() const;
  [[nodiscard]] std::vector<cudf::size_type> const& get_row_group_indices() const;
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const&
    get_column_chunk_byte_ranges() const;
  [[nodiscard]] std::size_t get_file_size() const;
  [[nodiscard]] std::shared_ptr<cudf::io::datasource> const& get_fallback_datasource() const;
  [[nodiscard]] std::shared_ptr<translated_expression> const& get_filter_expression() const;
  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const;

  void set_post_convert_fn(post_convert_fn_t fn);
  [[nodiscard]] bool has_post_convert_fn() const;
  [[nodiscard]] std::unique_ptr<cudf::table> apply_post_convert(
    std::unique_ptr<cudf::table> tbl, rmm::cuda_stream_view stream);

 private:
  rmm::device_buffer _column_chunks;
  std::shared_ptr<hybrid_scan_reader> _parquet_reader;
  cudf::io::parquet_reader_options _reader_options;
  std::vector<cudf::size_type> _row_group_indices;
  std::vector<cudf::io::text::byte_range_info> _column_chunk_byte_ranges;
  std::size_t _size_in_bytes;
  std::size_t _uncompressed_size_in_bytes;
  std::size_t _file_size;
  std::shared_ptr<cudf::io::datasource> _fallback_datasource;
  std::shared_ptr<translated_expression> _filter_expression;
  std::vector<std::size_t> _post_filter_projection_ids;
  post_convert_fn_t _post_convert_fn;
};

}  // namespace sirius::data
```

### Verify

Build unchanged — headers are not included by any existing code. Regression passes trivially.

---

## PR 2: Datasource Factory + Local File Path -- DONE

**Status:** Committed. Build passes.

### Changes

**New files:**
- `src/op/scan/datasource_factory.cpp` — implement `create()` for local paths and `file://` scheme. S3 URIs throw `std::runtime_error`.
- `test/cpp/scan/test_datasource_factory.cpp` — tests for local paths, `file://`, error on `s3://`.

**Modified files:**
- `CMakeLists.txt` — add `src/op/scan/datasource_factory.cpp` to `EXTENSION_SOURCES`, add test file to `TEST_SOURCES`.
- `src/op/scan/parquet_scan_task.cpp` — replace call sites at lines 265 and 491 with `datasource_factory::create(file_path)`.
- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` — replace call site at line 251.

### Tests

```cpp
TEST_CASE("datasource_factory local file", "[datasource]") { ... }
TEST_CASE("datasource_factory file:// scheme", "[datasource]") { ... }
TEST_CASE("datasource_factory throws on s3:// uri", "[datasource]") { ... }
TEST_CASE("datasource_factory empty path throws", "[datasource]") { ... }
```

### Verify

Build, `sirius_unittest "[datasource]"`, full regression. All existing `[parquet_scan_task]` tests pass unchanged.

---

## PR 3: S3 Datasource (KvikIO RemoteHandle) -- IN PROGRESS

**Status:** Code complete. Build blocked by network issue (Catch2 FetchContent git clone fails through proxy after `rm -rf build`). See [pr3-build-note.md](pr3-build-note.md). Needs incremental build from existing build dir to verify.

### Build system changes (colocated)

- `vcpkg_ports/kvikio/portfile.cmake` line 77: `OFF` → `ON` (enable KvikIO remote, adds libcurl 8.13.0 static).
- `vcpkg_ports/cudf/portfile.cmake` line 155: `OFF` → `ON`.

### Config infrastructure (colocated)

- `src/include/sirius_config.hpp` — add `#include "op/scan/object_store_config.hpp"`, add `object_store_config` member + getter/setter.
- `src/sirius_config.cpp` — add `object_store:` YAML section parsing using `yaml::reader`.
- `src/sirius_extension.cpp` — register DuckDB `SET` commands: `s3_endpoint`, `s3_region`, `s3_access_key_id`, `s3_secret_access_key`, `s3_transport`, `s3_use_tls`.

### S3 datasource implementation

**New files:**
- `src/op/scan/s3_datasource.cpp` — implement pimpl using `kvikio::RemoteHandle`.
- `test/cpp/config/test_object_store_config.cpp` — config parsing tests.
- `test/cpp/scan/test_s3_datasource.cpp` — S3 datasource tests.

**Modified files:**
- `CMakeLists.txt` — add new source and test files.
- `src/op/scan/datasource_factory.cpp` — wire S3 URI dispatch to `s3_datasource`. Add config plumbing: obtain `object_store_config` from `sirius_config` at call sites.
- `src/op/scan/parquet_scan_task.cpp` — pass `object_store_config` to factory at both call sites.
- `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` — same.

### Tests

```cpp
// Config tests
TEST_CASE("object_store_config defaults", "[config_opt][datasource]") { ... }
TEST_CASE("object_store_config yaml parsing", "[config_opt][datasource]") { ... }
TEST_CASE("object_store_config yaml rejects unknown keys", "[config_opt][datasource]") { ... }
TEST_CASE("object_store_config transport enum", "[config_opt][datasource]") { ... }

// S3 datasource tests (skip without network / credentials)
TEST_CASE("s3_datasource size via http", "[datasource][s3]") { ... }
TEST_CASE("s3_datasource host_read range", "[datasource][s3]") { ... }
TEST_CASE("s3_datasource host_read_async", "[datasource][s3]") { ... }
TEST_CASE("factory creates s3_datasource for s3:// uri", "[datasource][s3]") { ... }
```

### Verify

Clean rebuild (vcpkg portfile changed). `sirius_unittest "[datasource]"`, full regression.

---

## PR 4: GDS Datasource (KvikIO FileHandle) -- DONE

**Status:** Committed. Build requires incremental build (network blocked for clean build).

### Build system change (colocated)

- `vcpkg_ports/kvikio/portfile.cmake` — remove the two `vcpkg_replace_string()` calls (lines 42-66) that disable cuFile. KvikIO's `try_compile()` will find the Batch/Stream APIs in cuFile 1.16+.

### GDS datasource implementation

**New files:**
- `src/op/scan/gds_datasource.cpp` — implement pimpl using `kvikio::FileHandle`.
- `test/cpp/scan/test_gds_datasource.cpp` — GDS datasource tests.

**Modified files:**
- `CMakeLists.txt` — add new source and test files.
- `src/op/scan/datasource_factory.cpp` — for local paths, probe `kvikio::is_cufile_available()`. If true, return `gds_datasource`; otherwise, fall through to `cudf::io::datasource::create()`.

### Tests

```cpp
TEST_CASE("gds_datasource falls back to posix without gds hw", "[datasource][gds]") { ... }
TEST_CASE("gds_datasource host_read matches cudf datasource", "[datasource][gds]") { ... }
TEST_CASE("gds_datasource size matches file size", "[datasource][gds]") { ... }
TEST_CASE("gds_datasource device_read_async with gds", "[datasource][gds]") {
  // Skip if no GDS hardware
}
```

### Verify

Clean rebuild (portfile changed). `sirius_unittest "[datasource]"`, full regression. Without GDS hardware, factory still returns `cudf::io::datasource` — existing behavior preserved.

---

## PR 5: GPU Parquet Representation + Converter

**Goal:** Add the new data type for compressed Parquet bytes in GPU memory and the converter to decompress them in-place on GPU.

### Implementation

**New files:**
- `src/data/gpu_parquet_representation.cpp` — accessors, `clone()` (deep-copy device buffer).
- `src/data/gpu_parquet_representation_converters.cpp` — converter: wrap GPU buffer in a `cudf::io::datasource`, call `cudf::io::read_parquet()` for in-place GPU decompression, apply post-convert hook, return `gpu_table_representation`.
- `test/cpp/data/test_gpu_parquet_representation.cpp`

**Modified files:**
- `CMakeLists.txt` — add new source and test files.
- `src/data/host_parquet_representation_converters.cpp` — register `gpu_parquet_representation → gpu_table_representation` converter in `register_parquet_converters()`.

### Tests

```cpp
TEST_CASE("gpu_parquet_representation size tracking", "[datasource][gpu_parquet]") { ... }
TEST_CASE("gpu_parquet_representation clone", "[datasource][gpu_parquet]") { ... }
TEST_CASE("gpu_parquet to gpu_table conversion", "[datasource][gpu_parquet]") {
  // Write small parquet, read bytes into GPU buffer, convert, verify row/col counts
}
```

### Verify

Build, `sirius_unittest "[datasource]"`, full regression.

---

## PR 6: Scan Task GPU Direct Branching

**Goal:** Wire the GDS datasource + `gpu_parquet_representation` end-to-end by branching in `parquet_scan_task::compute_task()`.

### Changes

**Modified files:**
- `src/op/scan/parquet_scan_task.cpp` — in `compute_task()`, after creating the datasource, branch:
  - `datasource->is_device_read_preferred(total_bytes)` → allocate `rmm::device_buffer`, `device_read_async()` each byte range into GPU, create `gpu_parquet_representation`, return as `data_batch`.
  - Otherwise → existing host path unchanged.

### Tests

Extend `test/cpp/scan/test_parquet_scan_task.cpp`:

```cpp
TEST_CASE("parquet_scan_task gpu direct path", "[shared_context][datasource][gds]") {
  // Skip if no GDS hardware
  // Write parquet, scan via GPU Direct, verify results match CPU path
}
TEST_CASE("parquet_scan_task host path still works", "[shared_context][datasource]") {
  // Regression: existing host path unaffected
}
```

### Verify

Build, `sirius_unittest "[datasource]"`, `sirius_unittest "[parquet_scan_task]"`, full regression.

---

## PR 7: RDMA S3 Datasource (cuObjClient)

**Goal:** Enable GPU Direct RDMA for S3-compatible stores (VAST, MinIO) using NVIDIA's cuObjClient library for the RDMA data plane and libcurl + SigV4 for the HTTP control plane.

### Build system changes (colocated)

- `vcpkg.json` — add `"curl"` to dependencies (needed for HTTP control path with custom `x-amz-rdma-token` headers; KvikIO remote's internal libcurl is not exported).
- `CMakeLists.txt`:
  - Add optional RDMA library discovery:
    ```cmake
    pkg_check_modules(LIBIBVERBS IMPORTED_TARGET libibverbs)
    pkg_check_modules(LIBRDMACM IMPORTED_TARGET librdmacm)
    ```
  - Find cuObjClient from the CUDA toolkit:
    ```cmake
    find_path(CUOBJCLIENT_INCLUDE_DIR cuobjclient.h
      HINTS ${CUDAToolkit_INCLUDE_DIRS}
            /usr/local/cuda/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/include)
    find_library(CUOBJCLIENT_LIBRARY cuobjclient
      HINTS ${CUDAToolkit_LIBRARY_DIR}
            /usr/local/cuda/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/lib)
    if(CUOBJCLIENT_LIBRARY AND CUOBJCLIENT_INCLUDE_DIR AND LIBIBVERBS_FOUND AND LIBRDMACM_FOUND)
      set(SIRIUS_RDMA_SUPPORT ON)
    endif()
    ```
  - Conditionally link `CURL::libcurl`, `${CUOBJCLIENT_LIBRARY}`, `PkgConfig::LIBIBVERBS`, `PkgConfig::LIBRDMACM`.
  - Define `SIRIUS_RDMA_SUPPORT` compile definition.

### RDMA S3 datasource implementation

**New files:**
- `src/op/scan/rdma_s3_datasource.cpp` — guarded by `#ifdef SIRIUS_RDMA_SUPPORT`. Architecture:
  - **RDMA data plane (cuObjClient):** Manages memory registration, DC transport, and RDMA token lifecycle.
    - Constructor: create `cuObjClient` with GET/PUT callback ops.
    - `device_read_async()`: register GPU buffer via `cuMemObjGetDescriptor()`, call `cuObjGet()` (wrapped in `std::async` to satisfy the async `std::future` interface), deregister via `cuMemObjPutDescriptor()`.
  - **HTTP control plane (libcurl + SigV4):** The GET callback receives the RDMA token from cuObjClient and sends an HTTP Range GET to the S3 endpoint with `x-amz-rdma-token` header + SigV4 auth. ~200 lines of self-contained SigV4 signing using libcurl + OpenSSL HMAC.
  - **`host_read()`:** HTTP Range GET fallback (no RDMA) for small metadata reads (footer, PAR1 header). Uses libcurl directly.
  - **`size()`:** HTTP HEAD request, cached.
- `src/op/scan/s3_sigv4.hpp` / `src/op/scan/s3_sigv4.cpp` — self-contained AWS SigV4 signing module (~200 lines). Shared between `rdma_s3_datasource` HTTP callbacks and potential future direct-libcurl S3 usage.
- `test/cpp/scan/test_rdma_s3_datasource.cpp`

**Modified files:**
- `CMakeLists.txt` — add new source and test files.
- `src/op/scan/datasource_factory.cpp` — for S3 URIs when `config.transport == RDMA` (or `AUTO` with cuObjClient + RDMA hardware detected), return `rdma_s3_datasource`.

### Key design notes

1. **Async wrapper for synchronous API.** `cuObjGet()` is synchronous. Wrap in `std::async(std::launch::async, ...)` to return `std::future<size_t>` as required by `cudf::io::datasource::device_read_async()`.

2. **4 GiB per-registration limit.** `CUOBJ_MAX_MEMORY_REG_SIZE` is 4 GiB. Default scan batch size is 512 MB — well within limit. Enforce with a check before `cuMemObjGetDescriptor()`.

3. **cuFile.h header dependency.** `cuobjclient.h` includes `cufile.h`. The cuFile patch must be removed (PR 4) before this PR to ensure clean header parsing.

4. **Server-side requirement.** The S3-compatible object store must integrate the cuObjServer library (VAST Data, MinIO with RDMA extensions). Standard cloud S3 (AWS, GCS, Azure) falls back to the HTTP-only `s3_datasource` (PR 3).

### Tests

```cpp
#ifdef SIRIUS_RDMA_SUPPORT
TEST_CASE("rdma_s3_datasource cuobjclient init", "[datasource][rdma]") {
  // Verify cuObjClient construction succeeds with callback ops
}
TEST_CASE("rdma_s3_datasource memory registration", "[datasource][rdma]") {
  // Allocate GPU buffer, register via cuMemObjGetDescriptor, verify success
  // Deregister via cuMemObjPutDescriptor
}
TEST_CASE("rdma_s3_datasource rdma token generation", "[datasource][rdma]") {
  // Register buffer, get RDMA token via cuMemObjGetRDMAToken
  // Verify token string is non-empty, free via cuMemObjPutRDMAToken
}
TEST_CASE("rdma_s3_datasource host_read via http", "[datasource][rdma][s3]") {
  // Verify host_read falls back to HTTP Range GET correctly
}
TEST_CASE("rdma_s3_datasource device_read_async", "[datasource][rdma][s3]") {
  // Full integration: requires RDMA-capable S3 endpoint (e.g., MinIO with cuObjServer)
  // Skip in CI without proper setup
}
#endif
```

### Verify

Build with and without RDMA libs + cuObjClient. `sirius_unittest "[datasource]"`, full regression. Systems without RDMA libs or cuObjClient: `SIRIUS_RDMA_SUPPORT` not defined, RDMA code compiled out, zero impact.

---

## PR 8: Iceberg S3 Support

**Goal:** Fix the 5 direct-I/O sites in the Iceberg pipeline so Iceberg-over-S3 works end-to-end, including V2 deletes and manifest resolution.

### Scope Assessment

There are 5 call sites that bypass the datasource factory:

| Site | File | I/O Method | Complexity |
|------|------|------------|------------|
| 1 | `iceberg_avro_reader.cpp:552` | `std::ifstream` | Medium — needs byte-buffer abstraction for Avro parsing |
| 2 | `iceberg_avro_reader.cpp:631` | `std::ifstream` | Same as above |
| 3 | `iceberg_scan_task.cpp:57` | `cudf::io::read_parquet(source_info{path})` | Low — replace `source_info{path}` with factory-created datasource |
| 4 | `iceberg_scan_task.cpp:119` | `cudf::io::read_parquet(source_info{path})` | Same as above |
| 5 | `iceberg_metadata_reader.cpp` | Orchestrator calling 1-4 | Plumbing only |

### Approach

**Sites 3-5** (delete file reads + orchestrator): replace the path-based `source_info{path}` with a datasource pointer from the factory via `source_info{datasource_ptr}`. Straightforward substitution.

**Sites 1-2** (Avro manifest reading via `std::ifstream`): replace `std::ifstream` with a factory-created datasource, read the entire file into a `std::vector<uint8_t>` via `datasource->host_read(0, datasource->size())`, then parse from the memory buffer. Avro manifests are small (typically < 1 MB), so reading the whole file into memory is safe and avoids introducing a new adapter class.

### Changes

**Modified files:**
- `src/op/scan/iceberg_avro_reader.cpp` — replace `std::ifstream` at lines 552 and 631 with `datasource_factory::create(path, config)` → `host_read(0, size)` → parse from `std::vector<uint8_t>` buffer.
- `src/op/scan/iceberg_scan_task.cpp` — replace `cudf::io::read_parquet(source_info{path})` at lines 57 and 119 with factory-created datasource passed via `source_info{datasource_ptr}`.
- `src/op/scan/iceberg_metadata_reader.cpp` — plumb `object_store_config` through to the above call sites.

### Tests

Extend existing Iceberg integration tests or add new ones:

```cpp
TEST_CASE("iceberg delete files via datasource factory", "[datasource][iceberg]") {
  // Create local iceberg table with positional deletes
  // Scan via datasource factory path, verify delete application
}
TEST_CASE("iceberg avro manifest read from buffer", "[datasource][iceberg]") {
  // Read an Avro manifest file via factory into buffer
  // Parse and verify manifest entries match direct ifstream parse
}
```

### Verify

Build, `sirius_unittest "[datasource]"`, `sirius_unittest "[integration]"`, full regression.

---

## PR 9: Documentation

**Modified files:**
- `docs/super-sirius/scan.md` — add "Storage Backends" section describing the datasource factory and supported backends.
- `docs/datasource/gap.md` — update Iceberg section to reflect PR 8 (no longer a limitation).
- `CLAUDE.md` — add datasource factory to Architecture section, note new build dependencies.

---

## Summary

### PR Flow

| PR | Description | Build/Config Changes | Key Property |
|----|-------------|---------------------|--------------|
| **1** | All interface headers | None | Review-only: full scope visible |
| **2** | Factory + local file path | `CMakeLists.txt` (new source) | Zero behavior change |
| **3** | S3 datasource + config | KvikIO remote ON, `sirius_config`, DuckDB SET | Unlocks `s3://` queries |
| **4** | GDS datasource | cuFile patch removed | Unlocks GPU Direct Storage |
| **5** | `gpu_parquet_representation` + converter | `CMakeLists.txt` (new sources) | New data type, no scan changes yet |
| **6** | Scan task GPU Direct branching | None | Activates GDS end-to-end |
| **7** | RDMA S3 datasource | `vcpkg.json` (curl), cuObjClient, RDMA libs | Unlocks RDMA S3 path (VAST, MinIO) |
| **8** | Iceberg S3 (5 I/O sites) | None | Full Iceberg-over-S3 support |
| **9** | Documentation | None | Docs only |

### New Files

| File | PR | Purpose |
|------|-----|---------|
| `src/include/op/scan/object_store_config.hpp` | 1 | Config struct |
| `src/include/op/scan/datasource_factory.hpp` | 1 | Factory interface |
| `src/include/op/scan/s3_datasource.hpp` | 1 | S3 datasource interface |
| `src/include/op/scan/gds_datasource.hpp` | 1 | GDS datasource interface |
| `src/include/op/scan/rdma_s3_datasource.hpp` | 1 | RDMA S3 datasource interface |
| `src/include/data/gpu_parquet_representation.hpp` | 1 | GPU-resident compressed parquet |
| `src/op/scan/datasource_factory.cpp` | 2 | Factory implementation |
| `src/op/scan/s3_datasource.cpp` | 3 | S3 implementation (KvikIO RemoteHandle) |
| `src/op/scan/gds_datasource.cpp` | 4 | GDS implementation (KvikIO FileHandle) |
| `src/data/gpu_parquet_representation.cpp` | 5 | GPU parquet implementation |
| `src/data/gpu_parquet_representation_converters.cpp` | 5 | GPU parquet → GPU table converter |
| `src/op/scan/rdma_s3_datasource.cpp` | 7 | RDMA S3 implementation (cuObjClient + libcurl) |
| `src/op/scan/s3_sigv4.hpp` | 7 | SigV4 signing interface |
| `src/op/scan/s3_sigv4.cpp` | 7 | SigV4 signing implementation (~200 lines) |
| `test/cpp/scan/test_datasource_factory.cpp` | 2 | Factory tests |
| `test/cpp/config/test_object_store_config.cpp` | 3 | Config tests |
| `test/cpp/scan/test_s3_datasource.cpp` | 3 | S3 tests |
| `test/cpp/scan/test_gds_datasource.cpp` | 4 | GDS tests |
| `test/cpp/data/test_gpu_parquet_representation.cpp` | 5 | GPU parquet tests |
| `test/cpp/scan/test_rdma_s3_datasource.cpp` | 7 | RDMA tests |

### Modified Files

| File | PRs | Change |
|------|-----|--------|
| `CMakeLists.txt` | 2-7 | New sources, test files, optional RDMA/curl deps |
| `vcpkg_ports/kvikio/portfile.cmake` | 3, 4 | Enable remote (PR 3), remove cuFile patch (PR 4) |
| `vcpkg_ports/cudf/portfile.cmake` | 3 | Enable KvikIO remote IO |
| `vcpkg.json` | 7 | Add `curl` dependency |
| `src/include/sirius_config.hpp` | 3 | Add `object_store_config` member |
| `src/sirius_config.cpp` | 3 | YAML parsing for `object_store:` |
| `src/sirius_extension.cpp` | 3 | DuckDB SET commands for S3 |
| `src/op/scan/parquet_scan_task.cpp` | 2, 3, 6 | Replace 2 call sites (PR 2), config plumbing (PR 3), GPU Direct branch (PR 6) |
| `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | 2, 3 | Replace 1 call site (PR 2), config plumbing (PR 3) |
| `src/data/host_parquet_representation_converters.cpp` | 5 | Register GPU parquet converter |
| `src/op/scan/iceberg_avro_reader.cpp` | 8 | Replace `std::ifstream` with factory-based reads |
| `src/op/scan/iceberg_scan_task.cpp` | 8 | Replace direct `read_parquet(path)` with factory |
| `src/op/scan/iceberg_metadata_reader.cpp` | 8 | Plumb config to Iceberg I/O sites |
