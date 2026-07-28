# Provides the INTERFACE target `sirius_lance_ffi`: the pinned lance-duckdb Rust
# staticlib plus the link flags that keep its symbols out of the extension's
# dynamic symbol table.
#
# Included only when SIRIUS_ENABLE_LANCE_KNN is ON.
#
# Two modes, same resulting target, so the consumer wiring never branches:
#
#   -DSIRIUS_LANCE_FFI_LIB=/path/to/liblance_duckdb_ffi.a
#       Link a staticlib that was built elsewhere. The crate pulls ~680 crates
#       and takes on the order of 20 minutes to build cold, so this is the
#       practical choice for iterating; the caller owns pinning it.
#
#   (default)
#       Fetch lance-duckdb at SIRIUS_LANCE_FFI_GIT_TAG and build it with
#       Corrosion. Reproducible, and the only mode that enforces the pin.
#       Needs a Rust toolchain, protoc, and network on the first build.

set(SIRIUS_LANCE_FFI_GIT_REPOSITORY
    "https://github.com/lancedb/lance-duckdb.git"
    CACHE STRING "Git repository providing the lance_duckdb_ffi staticlib")

# Pinned deliberately to a commit rather than a branch: the C surface this shim
# declares in src/lance_shim/lance_embedded_ffi.cpp is not covered by any
# stability guarantee, and drift shows up as a link error at best.
set(SIRIUS_LANCE_FFI_GIT_TAG
    "6316cbb"
    CACHE STRING "lance-duckdb commit providing the pinned FFI surface")

set(SIRIUS_LANCE_FFI_LIB
    ""
    CACHE FILEPATH
    "Prebuilt liblance_duckdb_ffi.a to link instead of building the crate")

add_library(sirius_lance_ffi INTERFACE)

# Needed by the crate at link time; verified against the M0 probe's link line.
#
# NOTE for anyone chasing a math-function crash here: the archive does carry weak
# compiler_builtins definitions of ceil/floor/round/fabs (and the f variants), so
# it looks like an obvious libm-hijack suspect. It is not. A feature-OFF build,
# with the archive entirely out of the link, segfaults identically on
# round/floor/ceil/abs -- that is a separate, pre-existing defect in the GPU
# projection path. Do not "fix" it by reordering these libraries.
find_package(Threads REQUIRED)
target_link_libraries(sirius_lance_ffi INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)

if(SIRIUS_LANCE_FFI_LIB)
  if(NOT EXISTS "${SIRIUS_LANCE_FFI_LIB}")
    message(
      FATAL_ERROR
        "SIRIUS_LANCE_FFI_LIB points at a file that does not exist: ${SIRIUS_LANCE_FFI_LIB}"
    )
  endif()
  message(STATUS "Lance FFI: linking prebuilt staticlib ${SIRIUS_LANCE_FFI_LIB}")
  target_link_libraries(sirius_lance_ffi INTERFACE "${SIRIUS_LANCE_FFI_LIB}")
  # --exclude-libs matches on the archive's file name, so it has to follow
  # whatever was actually linked. Hardcoding liblance_duckdb_ffi.a here would
  # silently stop hiding the symbols the moment someone points
  # SIRIUS_LANCE_FFI_LIB at a differently named copy.
  get_filename_component(_lance_ffi_archive_name "${SIRIUS_LANCE_FFI_LIB}" NAME)
else()
  message(
    STATUS
      "Lance FFI: building lance-duckdb@${SIRIUS_LANCE_FFI_GIT_TAG} via Corrosion "
      "(cold build is slow; pass -DSIRIUS_LANCE_FFI_LIB=<path> to reuse one)")

  include(FetchContent)
  # Same Corrosion the telemetry bridge uses; FetchContent dedupes the second
  # declaration, so the versions must not diverge.
  FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.6.1)
  FetchContent_MakeAvailable(Corrosion)

  FetchContent_Declare(
    lance_duckdb
    GIT_REPOSITORY "${SIRIUS_LANCE_FFI_GIT_REPOSITORY}"
    GIT_TAG "${SIRIUS_LANCE_FFI_GIT_TAG}")
  FetchContent_MakeAvailable(lance_duckdb)

  # --locked: the pin is only meaningful if the dependency graph is pinned too.
  corrosion_import_crate(
    MANIFEST_PATH "${lance_duckdb_SOURCE_DIR}/Cargo.toml"
    CRATES lance_duckdb_ffi
    PROFILE release
    LOCKED)

  target_link_libraries(sirius_lance_ffi INTERFACE lance_duckdb_ffi)
  set(_lance_ffi_archive_name "liblance_duckdb_ffi.a")
endif()

# Keep every lance_* symbol private to the extension. Two reasons: the process
# may already host a real Lance extension with its own copy of these symbols,
# and the crate is built with panic="abort", so an accidental cross-binding
# would abort the process rather than surface an error. Scoped to this archive
# on purpose -- --exclude-libs,ALL would also hide the static archives whose
# symbols the extension is required to export.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(sirius_lance_ffi INTERFACE
                      "LINKER:--exclude-libs,${_lance_ffi_archive_name}")
endif()
