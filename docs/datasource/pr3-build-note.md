# PR 3: Build Note — Network Dependency on Clean Build

## Issue

When building from a clean `build/` directory (after `rm -rf build`), CMake's FetchContent
fetches Catch2 from `https://github.com/catchorg/Catch2.git`. This requires network access
during the CMake configure step.

The proxy (`192.168.0.106:8089`) is not automatically propagated to git clone commands run
by CMake's FetchContent. This causes the clone to fail with:

```
error: RPC failed; curl 56 Recv failure: Connection timed out
```

## Workaround

Avoid `rm -rf build` when iterating. The Catch2 source is cached in
`build/release/_deps/catch2-src/` after the first successful configure.

If a clean build is needed:
1. Pre-clone Catch2 to a local cache directory
2. Set `FETCHCONTENT_SOURCE_DIR_CATCH2=/path/to/catch2-src` in CMake

## Compilation Fix Applied

The `cufile.h` header was not found when directly including KvikIO headers
(`kvikio/remote_handle.hpp` → `kvikio/shim/cufile_h_wrapper.hpp` → `<cufile.h>`).

Fix: added `find_package(CUDAToolkit REQUIRED)` and
`target_include_directories(... PRIVATE ${CUDAToolkit_INCLUDE_DIRS})` to CMakeLists.txt.
This adds the pixi CUDA toolkit include path
(`/cpp/sirius/.pixi/envs/default/targets/x86_64-linux/include/`) where `cufile.h` lives.
