# Gap Analysis: Multi-Backend Datasource Layer

**Based on:** [sirius-multidatasource-proposal.md](sirius-multidatasource-proposal.md)
**Date:** 2026-04-13

This document identifies the implementation gaps between the proposal and the current Sirius codebase.

---

## 1. Datasource Creation Call Sites

The proposal identifies **2 call sites** in `parquet_scan_task.cpp`. The codebase actually has **3**:

| # | File | Line | Purpose | Covered by proposal? |
|---|------|------|---------|:-------------------:|
| 1 | `src/op/scan/parquet_scan_task.cpp` | 265 | Footer reading (global state init) | Yes |
| 2 | `src/op/scan/parquet_scan_task.cpp` | 491 | Per-task data reading (`compute_task`) | Yes |
| 3 | `src/op/scan/sirius_parquet_metadata_scan_operator.cpp` | 251 | Metadata scan footer reading | **No** |

Call site 3 is a separate operator (`sirius_parquet_metadata_scan_operator`) that also creates datasources via `cudf::io::datasource::create(file_path)` to read Parquet footers. The factory must cover this call site too, or it will also fail on S3 URIs.

---

## 2. Iceberg Out-of-Scope I/O Surface

The proposal acknowledges Iceberg delete/manifest reads are out of scope but does not quantify the surface. There are **5 additional direct-I/O call sites**, all of which will break on S3 URIs:

| File | Function | Line | I/O Method | Problem |
|------|----------|------|------------|---------|
| `src/op/scan/iceberg_avro_reader.cpp` | `read_iceberg_manifest_list()` | 552 | `std::ifstream` | Completely incompatible with S3 |
| `src/op/scan/iceberg_avro_reader.cpp` | `read_iceberg_manifest_delete_files()` | 631 | `std::ifstream` | Completely incompatible with S3 |
| `src/op/scan/iceberg_scan_task.cpp` | `read_positional_delete_file()` | 57 | `cudf::io::read_parquet(source_info{path})` | Passes path directly; no factory |
| `src/op/scan/iceberg_scan_task.cpp` | `read_equality_delete_file()` | 119 | `cudf::io::read_parquet(source_info{path})` | Passes path directly; no factory |
| `src/op/scan/iceberg_metadata_reader.cpp` | (orchestrator) | -- | Calls the above functions | Passes DuckDB-resolved paths |

### Call Chain

```
iceberg_metadata_reader::read_iceberg_delete_metadata()
  |-> iceberg_snapshots() [DuckDB] -> manifest_list_path (can be S3)
  |
  |-> iceberg_avro_reader::read_iceberg_manifest_list(manifest_list_path)
  |   \-> std::ifstream (line 552)  <-- BREAKS ON S3
  |       \-> Returns: vector<(manifest_path, content_type)>
  |
  \-> For each manifest_path:
      \-> iceberg_avro_reader::read_iceberg_manifest_delete_files(manifest_path)
          \-> std::ifstream (line 631)  <-- BREAKS ON S3
              \-> Returns: vector<delete_file_paths>
                  |
                  |-> read_positional_delete_file(delete_file_path)
                  |   \-> cudf::io::read_parquet() (line 57)  <-- BREAKS ON S3
                  |
                  \-> read_equality_delete_file(delete_file_path)
                      \-> cudf::io::read_parquet() (line 119)  <-- BREAKS ON S3
```

### Impact

Iceberg-over-S3 is fully supported after PR 8. All 5 direct-I/O sites now go through the datasource factory.

---

## 3. Configuration — No Existing S3/Object Store Infrastructure

There is **zero** S3, object-store, endpoint, credential, or transport configuration anywhere in the current codebase. Everything must be built from scratch.

### Current Config Systems

| System | Mechanism | Location |
|--------|-----------|----------|
| DuckDB-level (`duckdb::Config`) | `SET` SQL commands via `AddExtensionOption()` | `src/sirius_extension.cpp` |
| Sirius-level (`sirius::sirius_config`) | YAML file (`sirius.yaml`) | `src/sirius_config.cpp` |

### What Needs to Be Built

1. **`object_store_config` struct** in `src/include/sirius_config.hpp`:
   - Fields: `endpoint`, `region`, `access_key_id`, `secret_access_key`, `session_token`, `transport` (enum: `AUTO`/`HTTP`/`RDMA`), TLS toggle, connection/request timeouts.

2. **YAML parsing** — add `object_store:` section handler in `src/sirius_config.cpp` using the existing `yaml::reader` infrastructure (already supports byte-suffix parsing, validators, `reject_unknown()`).

3. **DuckDB `SET` commands** — register `s3_endpoint`, `s3_region`, `s3_access_key_id`, `s3_secret_access_key`, `s3_transport` via `AddExtensionOption()` in `src/sirius_extension.cpp` (20+ options already registered there as a pattern to follow).

4. **Threading config to scan tasks** — `parquet_scan_task_global_state` currently receives the `sirius_physical_parquet_scan` operator + `sirius_context`. The `sirius_config` is reachable via `sirius_context`, so the factory can get the config from there without new plumbing.

---

## 4. Build System — Dependencies Not Yet Present

| Dependency | Current Status | Required For | Action |
|------------|---------------|--------------|--------|
| **KvikIO** | Transitive dep via cudf; present in `vcpkg_ports/cudf/vcpkg.json` | `gds_datasource` | Already available; link headers |
| **KvikIO Remote** | **Explicitly disabled**: `KvikIO_REMOTE_SUPPORT=OFF` (`vcpkg_ports/kvikio/portfile.cmake:77`), `CUDF_KVIKIO_REMOTE_IO=OFF` (`vcpkg_ports/cudf/portfile.cmake:155`) | Alternative to libcurl for S3 | Flip flags (pulls in libcurl transitively) |
| **libcurl** | **Not present** anywhere in build | `s3_datasource` (HTTP Range GET + SigV4) | Add to `vcpkg.json` **OR** enable KvikIO remote |
| **cuFile SDK** | KvikIO portfile patches *out* cuFile Batch/Stream APIs | `gds_datasource` (GPU Direct Storage) | Patch is aggressive — needs review for compatibility |
| **RDMA libs** (libibverbs, librdmacm) | **Installed** (rdma-core 50.0) | `rdma_s3_datasource` | Available system-wide; link via pkg-config |
| **cuObjClient** (`libcuobjclient.so`) | Present in CUDA 13.2 toolkit (`/usr/local/cuda-13.2`), **not** in pixi env | `rdma_s3_datasource` | See "cuObjClient Analysis" below |

### Build Decision: libcurl vs KvikIO Remote

The proposal recommends **libcurl + SigV4 directly** over enabling KvikIO remote. Tradeoffs:

| Approach | Pros | Cons |
|----------|------|------|
| **Direct libcurl** | Simpler, self-contained, ~200 lines SigV4 | No GCS/Azure without more work |
| **KvikIO remote** | Consolidated; gets GCS/Azure/WebHDFS for free | Dependency toggle complexity, cuFile SDK version constraints |

### Actual Cost of Enabling KvikIO Remote (`KvikIO_REMOTE_SUPPORT=ON`)

Flipping the flag in `vcpkg_ports/kvikio/portfile.cmake:77` and `vcpkg_ports/cudf/portfile.cmake:155` introduces **one new dependency**: libcurl, built from source as a static library.

**What gets added:**

| Component | Size / Lines | Notes |
|-----------|-------------|-------|
| **libcurl 8.13.0** (`libcurl.a`, static) | ~1-1.5 MB compiled | Fetched via CPM from `curl/curl` GitHub, built with minimal options: no CLI exe, no LDAP, no PSL, no shared lib |
| **OpenSSL** (TLS backend for libcurl) | **0 additional** | Already present on the system (`libssl3` 3.0.13) |
| **zlib** (HTTP compression) | **0 additional** | Already in the build (cudf dependency) |

libcurl is linked into `libkvikio.so` with `--exclude-libs,ALL`, so its symbols are **not exported** and will not conflict with any other libcurl on the system.

**KvikIO remote source files added (6 files, ~1,600 lines total):**

| File | Lines | Purpose |
|------|-------|---------|
| `src/remote_handle.cpp` | 827 | S3 (SigV4 via env vars), HTTP/HTTPS, presigned URL endpoint handling |
| `src/detail/url.cpp` | 284 | URL parsing and validation |
| `src/shim/libcurl.cpp` | 186 | libcurl dlopen shim (lazy loading) |
| `src/hdfs.cpp` | 140 | WebHDFS endpoint support |
| `src/detail/tls.cpp` | 139 | TLS certificate path resolution |
| `src/detail/remote_handle.cpp` | 21 | Detail forwarding |

**What you get for free:** `kvikio::RemoteHandle` with built-in support for S3, GCS, HTTP/HTTPS, WebHDFS, and presigned URLs — all implementing a `cudf::io::datasource`-compatible read interface. This would eliminate the need to write a custom `s3_datasource` (~500-800 lines) and SigV4 signing code (~200 lines).

**libcurl build options** (set by KvikIO's `get_libcurl.cmake`):

```cmake
BUILD_CURL_EXE OFF             # No curl CLI binary
BUILD_SHARED_LIBS OFF          # Static only (linked into libkvikio.so)
BUILD_TESTING OFF              # No tests
CURL_USE_LIBPSL OFF            # No public suffix list
CURL_DISABLE_LDAP ON           # No LDAP
CMAKE_POSITION_INDEPENDENT_CODE ON
EXCLUDE_FROM_ALL YES           # Not installed separately
```

### KvikIO cuFile Patch — Review Results

The vcpkg portfile for KvikIO (`vcpkg_ports/kvikio/portfile.cmake`) applies two patches. Here is what they do and whether they block the `gds_datasource` design.

#### Patch 1: Force `cuFile_FOUND = 0` (line 42-44)

```cmake
# Original:
if(NOT TARGET CUDA::cuFile)
# Patched to:
if(TRUE) # Disable cuFile/GDS - batch/stream API requires newer cuFile SDK
```

This forces the CMake branch that sets `cuFile_FOUND = 0`, which means:
- `KVIKIO_CUFILE_FOUND` is **not defined** as a compile definition
- All `#ifdef KVIKIO_CUFILE_FOUND` blocks compile to their `#else` stubs
- `cuFileAPI::cuFileAPI()` becomes `{ KVIKIO_FAIL("KvikIO not compiled with cuFile.h"); }`
- `is_cufile_library_available()` becomes `constexpr false`
- `cufile_version()` returns `0`

**Effect on `FileHandle`:** `CompatModeManager` sets `_is_compat_mode_preferred = true` (because `defaults::is_compat_mode_preferred()` returns true when cuFile is unavailable). Every `FileHandle::read()` and `FileHandle::pread()` call takes the **POSIX fallback path** (`detail::posix_device_read`), which reads into host memory via `pread()` then copies to GPU via `cudaMemcpy`. GDS (cuFile DMA) is never used.

#### Patch 2: Stub declarations for Batch/Stream types (lines 49-66)

Adds stub type definitions (`CUfileBatchHandle_t`, `CUfileOpcode_t`, `CUfileIOParams_t`, etc.) and function signatures so that the `cuFileAPI` shim class compiles — it uses `decltype()` on these symbols to declare its function pointer members. Without the stubs, the `decltype(cuFileBatchIOSetUp)` in `cufile.hpp` would fail to compile because the Batch/Stream declarations don't exist without `cufile.h`.

These stubs are never called at runtime. They only satisfy the compiler.

#### Verdict: The Patch Completely Blocks GDS

With the current patch, `kvikio::FileHandle` is a **POSIX-only** wrapper. It opens files, reads via `pread()`, and copies to GPU via `cudaMemcpy`. The cuFile DMA path is compiled out. A `gds_datasource` wrapping this `FileHandle` would get zero GDS benefit — it would behave identically to the existing `cudf::io::datasource::create(path)`.

#### Fix: Remove the Patch (Safe on This System)

The patch exists because the Batch/Stream APIs (`cuFileBatchIOSetUp`, `cuFileReadAsync`, `cuFileStreamRegister`, etc.) were only added in cuFile 1.7 (CUDA 12.2). The portfile was being conservative for older CUDA toolkits.

**Current system state:**
- Pixi environment: `libcufile-1.16.1` (CUDA 13 conda package) — has all Batch/Stream APIs
- System CUDA: `/usr/local/cuda-13.2` with `libcufile.so.1.17.1` — has all Batch/Stream APIs
- `cufile.h` in both locations contains `cuFileBatchIOSetUp`, `cuFileReadAsync`, `cuFileWriteAsync`, `cuFileStreamRegister`, `cuFileStreamDeregister`, `cuFileGetVersion`

Both environments have cuFile >= 1.7 with full Batch/Stream API support. The original `try_compile()` checks in KvikIO's CMakeLists.txt would pass. **Removing the patch is safe** on this system.

To re-enable GDS, remove both `vcpkg_replace_string()` calls (lines 42-66) from `vcpkg_ports/kvikio/portfile.cmake`. KvikIO's own CMake will then:
1. Find `CUDA::cuFile` target
2. `try_compile()` Batch and Stream APIs — both succeed
3. Set `cuFile_FOUND = 1`, define `KVIKIO_CUFILE_FOUND`
4. `cuFileAPI` constructor loads `libcufile.so.0` via `dlopen` and resolves all symbols
5. `FileHandle::read()` uses cuFile DMA when `compat_mode == AUTO` or `OFF`

**Portability note:** If the build must also work on systems with CUDA < 12.2 (no Batch/Stream API), the patch should be replaced with a conditional that checks cuFile version rather than disabling it unconditionally. KvikIO's own `try_compile()` already does this — the `FATAL_ERROR` on missing Batch API is the real problem. Upstream KvikIO chose to require it; the portfile worked around this by disabling cuFile entirely. A less aggressive fix would be to downgrade `FATAL_ERROR` to `WARNING` and set `cuFile_FOUND = 0` only when the APIs are actually missing.

### cuObjClient Analysis (RDMA S3 Simplification)

The proposal's `rdma_s3_datasource` design described raw libibverbs usage (`ibv_reg_mr`, DC QP management, RDMA token construction). NVIDIA's **cuObjClient** library (part of cuObject / GPUDirect Storage for Objects, shipped in CUDA 13.1.1+) encapsulates all of this and should be used instead.

#### What cuObjClient Provides

cuObjClient is a callback-based library that manages:
- **RDMA memory registration:** `cuMemObjGetDescriptor(ptr, size)` handles `ibv_reg_mr` for system, CUDA managed, and CUDA device memory (max 4 GiB per registration).
- **RDMA token generation:** `cuMemObjGetRDMAToken(ptr, size, offset, op, &token)` produces the `x-amz-rdma-token` string for the HTTP header. No manual rkey/vaddr/GID construction.
- **DC transport:** Uses `CUOBJ_PROTO_RDMA_DC_V1` — Dynamically Connected transport over InfiniBand or RoCEv2. No pre-established connections needed.
- **Chunked large transfers:** If a request exceeds `MaxRequestCallbackSize`, the library invokes the callback multiple times with successive chunks.

The I/O flow for a GET is:
1. Register GPU buffer: `cuMemObjGetDescriptor(d_ptr, size)`
2. Call `cuObjGet(ctx, d_ptr, size)` — this invokes the user-provided GET callback with the RDMA token
3. In the callback: send an HTTP GET to the S3 endpoint with `x-amz-rdma-token` header. The server calls `cuObjServer::handleGetObject()` which RDMA_WRITEs data to the client GPU buffer.
4. HTTP 200 OK returns, callback returns bytes read
5. Deregister: `cuMemObjPutDescriptor(d_ptr)`

#### What We Still Need to Write

cuObjClient handles the RDMA plumbing but **not the HTTP/S3 control path**. The user callback must:
1. Construct the S3 GET request with proper headers (`Range`, `Authorization` via SigV4, `x-amz-rdma-token`)
2. Send it to the S3 endpoint via HTTP
3. Parse the response (status code, `x-amz-rdma-reply` header)

This means `rdma_s3_datasource` still needs **libcurl + SigV4** for the control plane, but the RDMA data plane (memory registration, DC transport, token format) is fully handled by cuObjClient.

#### What cuObjClient Eliminates

| Component | Proposal (raw libibverbs) | With cuObjClient |
|-----------|--------------------------|------------------|
| RDMA memory registration | Manual `ibv_reg_mr()` + MR lifetime management | `cuMemObjGetDescriptor()` / `cuMemObjPutDescriptor()` |
| RDMA token construction | Manual rkey/vaddr/GID serialization | `cuMemObjGetRDMAToken()` → opaque string |
| DC QP management | Manual `ibv_create_qp()`, connection establishment | Handled internally by cuObjClient |
| Multi-NIC selection | Manual `ibv_get_device_list()` + selection logic | Handled internally (optimal NIC selection) |
| Chunked large reads | Manual splitting + offset tracking | Automatic via `MaxRequestCallbackSize` + multi-callback |
| GPU memory type detection | Manual `cudaPointerGetAttributes()` | `cuObjClient::getMemoryType()` |

Lines of code saved: ~500-700 lines of raw RDMA plumbing reduced to ~50 lines of cuObjClient API calls + the HTTP callback (~200 lines for libcurl + SigV4).

#### Availability and Dependencies

**Current system:**
- `libcuobjclient.so.1.1.1` present at `/usr/local/cuda-13.2/targets/x86_64-linux/lib/`
- `cuobjclient.h` at `/usr/local/cuda-13.2/targets/x86_64-linux/include/`
- Runtime deps: `libcufile.so.0`, `libibverbs.so.1`, `librdmacm.so.1`, `libmlx5.so.1` — all installed

**Pixi environment:** cuObjClient is **not** in the pixi conda packages (only `libcufile` is). The system CUDA toolkit provides it.

**Header dependency:** `cuobjclient.h` includes `cufile.h`. This means the cuFile patch in the KvikIO portfile (which we're removing in PR 4) doesn't directly affect cuObjClient — cuObjClient links to `libcufile.so` at the system level, not through KvikIO.

#### New Problems Introduced

1. **Build environment gap.** cuObjClient is in the system CUDA toolkit (`/usr/local/cuda-13.2`) but not in the pixi conda environment. The build must find `cuobjclient.h` and link `libcuobjclient.so` from the system toolkit path. This requires either:
   - Adding `-I/usr/local/cuda-13.2/targets/x86_64-linux/include` and `-L/usr/local/cuda-13.2/targets/x86_64-linux/lib` to the CMake config, or
   - Using `find_library(CUOBJCLIENT cuobjclient HINTS ...)` with a CUDA toolkit path hint, or
   - Requesting that cuObjClient be packaged in the pixi/conda environment (upstream to NVIDIA conda channel).

2. **CUDA 13.1.1+ requirement.** cuObjClient was introduced in CUDA 13.1.1. Systems with CUDA 12.x don't have it. The `rdma_s3_datasource` must be guarded by both `SIRIUS_RDMA_SUPPORT` and a cuObjClient availability check. Without cuObjClient, the RDMA S3 path is unavailable.

3. **cuFile.h header coupling.** `cuobjclient.h` includes `cufile.h`. If building in an environment where `cufile.h` is patched or unavailable (e.g., the old KvikIO portfile stubs), the header may not parse correctly. The cuFile patch removal (PR 4) should happen before or with the RDMA PR.

4. **Synchronous API.** `cuObjGet()` is synchronous — it blocks until the full transfer completes. The `cudf::io::datasource` interface expects `device_read_async()` returning a `std::future`. We need to wrap `cuObjGet()` in a `std::async()` call or a dedicated I/O thread to satisfy the async interface.

5. **4 GiB per-registration limit.** `CUOBJ_MAX_MEMORY_REG_SIZE` is 4 GiB. Parquet column chunks for a single scan batch are typically well under this (default batch size is 512 MB), but the limit should be documented and enforced.

6. **Server-side requirement.** cuObjClient only handles the client side. The S3-compatible object store must integrate the **cuObjServer** library to handle `x-amz-rdma-token` headers and perform RDMA_WRITE. Currently, VAST Data and MinIO (with RDMA extensions) support this protocol. AWS S3, GCS, and Azure do not.

---

## 5. New Code That Must Be Written

### 5a. `datasource_factory` (core integration)

- **New files:** `src/include/op/scan/datasource_factory.hpp`, `src/op/scan/datasource_factory.cpp`
- **Dispatch logic:** URI scheme (`file://`, `s3://`, bare path) + config (`s3_transport`) -> one of 4 datasource types
- **Integration:** Replace **3** `cudf::io::datasource::create()` call sites (not 2)

### 5b. `s3_datasource` (CPU S3 read path)

- **New files:** `src/include/op/scan/s3_datasource.hpp`, `src/op/scan/s3_datasource.cpp`
- **Implements:** `cudf::io::datasource` interface
- **Internals:** libcurl connection pool, HTTP HEAD for `size()`, HTTP Range GET for `host_read()`/`host_read_async()`, AWS SigV4 signing
- **Estimated complexity:** ~500-800 lines including connection management, retries, error handling

### 5c. `gds_datasource` (GPU Direct via KvikIO)

- **New files:** `src/include/op/scan/gds_datasource.hpp`, `src/op/scan/gds_datasource.cpp`
- **Implements:** `cudf::io::datasource` with `supports_device_read() = true`
- **Wraps:** `kvikio::FileHandle` for cuFile DMA
- **Fallback:** Auto-falls back to POSIX when GDS hardware/driver is absent (KvikIO handles this)
- **Estimated complexity:** ~200-300 lines

### 5d. `gpu_parquet_representation` (GPU Direct data type)

- **New files:** `src/include/data/gpu_parquet_representation.hpp`, `src/data/gpu_parquet_representation.cpp`
- **Mirrors:** `host_parquet_representation` but holds compressed bytes in GPU memory (not host)
- **Converter:** `gpu_parquet_representation -> gpu_table_representation` (in-place GPU decompression, no H2D copy)
- **Registration:** New converter in converter registry

There is currently **no** `gpu_parquet_representation` class. The system only supports:
- `host_parquet_representation` (compressed bytes in host memory)
- `gpu_table_representation` (decompressed `cudf::table` on GPU)

### 5e. Scan task branching for GPU Direct

- **File:** `src/op/scan/parquet_scan_task.cpp`, `compute_task()` function
- **Change:** Check `datasource->is_device_read_preferred()`, then branch:
  - **Yes:** `device_read_async()` -> GPU buffer -> `gpu_parquet_representation` -> in-place GPU decompress
  - **No:** existing `host_read_async()` -> host buffer -> `host_parquet_representation` path
- **Impact:** This is the most invasive change — it splits the scan task data flow into two paths

### 5f. `rdma_s3_datasource` (RDMA GPU Direct for S3)

- **New files:** `src/include/op/scan/rdma_s3_datasource.hpp`, `src/op/scan/rdma_s3_datasource.cpp`
- **Complexity:** Highest of all — requires RDMA memory registration, vendor-specific token headers, libibverbs integration
- **Likely deferred** to a later phase

---

## 6. What Stays Unchanged

The proposal's claim that most of the pipeline is untouched is validated by the codebase:

| Component | Changes? | Why |
|-----------|:--------:|-----|
| `prefetched_data_source` | No | Only sees `cudf::io::datasource` interface |
| `cache_ranges` | No | Only maps byte offsets to host blocks |
| `host_parquet_representation` | No | Storage-agnostic compressed data wrapper |
| `gpu_pipeline_task::compute_task()` | No | Operates on `gpu_table_representation` post-conversion |
| `sirius_pipeline_converter` | No | Pipeline topology unchanged |
| All GPU operators (filter, join, agg, etc.) | No | All operate on `cudf::table` |
| Iceberg post-convert hooks | No | Run after decompression regardless of source |
| `data_batch` / `shared_data_repository` | No | Storage-agnostic batch management |
| Converter registry | Extend only | Add `gpu_parquet_representation` converter in Phase 2 |

---

## 7. Implementation Phases

| Phase | Scope | New Files | Changed Files | Dependency |
|-------|-------|-----------|---------------|------------|
| **1: Factory + S3 CPU** | S3 read via HTTP Range GET | `datasource_factory.{hpp,cpp}`, `s3_datasource.{hpp,cpp}`, `object_store_config` | `parquet_scan_task.cpp` (2 sites), `sirius_parquet_metadata_scan_operator.cpp` (1 site), `sirius_config.{hpp,cpp}`, `sirius_extension.cpp` | Add libcurl to `vcpkg.json` |
| **2: GDS (Local NVMe)** | GPU Direct Storage for local files | `gds_datasource.{hpp,cpp}`, `gpu_parquet_representation.{hpp,cpp}`, new converter | `parquet_scan_task.cpp` (`compute_task` branch), converter registry | Review KvikIO cuFile patch |
| **3: RDMA S3** | GPU Direct for RDMA object stores | `rdma_s3_datasource.{hpp,cpp}` | Same as Phase 2 | libibverbs, vendor RDMA libs |
| **4: Iceberg S3** | Fix out-of-scope Iceberg I/O (5 sites) | Refactor existing files | `iceberg_avro_reader.cpp`, `iceberg_scan_task.cpp`, `iceberg_metadata_reader.cpp` | Phase 1 |

---

## 8. Key Corrections to the Proposal

1. **3 datasource call sites, not 2.** `sirius_parquet_metadata_scan_operator.cpp:251` is a third `cudf::io::datasource::create()` site that the factory must cover.

2. **5 Iceberg direct-I/O sites** will break on S3 (2x `std::ifstream` for Avro manifests, 2x `cudf::io::read_parquet` for delete files, 1x orchestrator). The proposal acknowledges these are out of scope but does not quantify the surface or note the user-facing limitation.

3. **Zero existing S3/object-store infrastructure.** Config structs, YAML parsing, DuckDB `SET` commands, and the libcurl dependency all need to be built from scratch.

4. **KvikIO cuFile patch completely blocks GDS.** The vcpkg portfile patches out cuFile entirely (`cuFile_FOUND = 0`), so `kvikio::FileHandle` is POSIX-only today. Removing the patch is safe on the current system (cuFile 1.16+/1.17+ with full Batch/Stream API). For portability to older CUDA toolkits, replace the unconditional disable with a graceful fallback.
