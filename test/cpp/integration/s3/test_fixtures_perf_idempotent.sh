#!/usr/bin/env bash
# Hermetic tests for fixtures.sh --perf reuse/idempotency decisions.
#
# These tests intentionally stub DuckDB, curl, docker, and the small-fixture
# Python generator so they do not require tpch, MinIO, Docker, or network access.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES_SH="${SCRIPT_DIR}/fixtures.sh"

FAILURES=0

fail() {
  echo "not ok - $*" >&2
  return 1
}

assert_file_exists() {
  [[ -e "$1" ]] || fail "expected file to exist: $1"
}

assert_file_not_exists() {
  [[ ! -e "$1" ]] || fail "expected file not to exist: $1"
}

assert_empty_or_missing() {
  [[ ! -s "$1" ]] || fail "expected empty/missing file: $1"
}

assert_nonempty() {
  [[ -s "$1" ]] || fail "expected non-empty file: $1"
}

assert_contains() {
  local file="$1"
  local needle="$2"
  grep -F -- "${needle}" "${file}" >/dev/null || fail "expected '${needle}' in ${file}"
}

assert_not_contains() {
  local file="$1"
  local needle="$2"
  if [[ -e "${file}" ]]; then
    ! grep -F -- "${needle}" "${file}" >/dev/null || fail "did not expect '${needle}' in ${file}"
  fi
}

assert_sha_valid() {
  local parquet="$1"
  (cd "$(dirname "${parquet}")" && sha256sum -c "$(basename "${parquet}").sha256" >/dev/null) ||
    fail "expected sha256 sidecar to validate for ${parquet}"
}

make_parquet() {
  local path="$1"
  local marker="${2:-cached}"
  mkdir -p "$(dirname "${path}")"
  printf 'PAR1%sPAR1' "${marker}" > "${path}"
}

write_sidecar() {
  local parquet="$1"
  (cd "$(dirname "${parquet}")" && sha256sum "$(basename "${parquet}")" > "$(basename "${parquet}").sha256")
}

make_stubs() {
  local root="$1"
  local duckdb_mode="$2"
  local bin="${root}/bin"
  mkdir -p "${bin}"

  cat > "${bin}/python3" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
out_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      out_dir="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -n "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi
exit 0
STUB

  cat > "${bin}/curl" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

  cat > "${bin}/docker" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${DOCKER_LOG:?}"
exit 0
STUB

  case "${duckdb_mode}" in
    success)
      cat > "${bin}/duckdb" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'duckdb %s\n' "$*" >> "${DUCKDB_CALL_LOG:?}"
cat >/dev/null
mkdir -p "${SIRIUS_BENCH_WORK_DIR:?}"
printf 'PAR1generated-%sPAR1' "$(date +%s%N)" > "${SIRIUS_BENCH_WORK_DIR}/lineitem_sf10.parquet"
exit 0
STUB
      ;;
    fail-if-called)
      cat > "${bin}/duckdb" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'duckdb %s\n' "$*" >> "${DUCKDB_CALL_LOG:?}"
cat >/dev/null || true
echo "duckdb stub must not be called" >&2
exit 42
STUB
      ;;
    fail)
      cat > "${bin}/duckdb" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'duckdb %s\n' "$*" >> "${DUCKDB_CALL_LOG:?}"
cat >/dev/null || true
echo "duckdb stub generation failure" >&2
exit 42
STUB
      ;;
    *)
      fail "unknown duckdb stub mode: ${duckdb_mode}"
      ;;
  esac

  chmod +x "${bin}/python3" "${bin}/curl" "${bin}/docker" "${bin}/duckdb"
}

run_fixtures() {
  local root="$1"
  local log="${root}/run.log"
  shift
  (
    export PATH="${root}/bin:${PATH}"
    export DUCKDB="${root}/bin/duckdb"
    export DOCKER_LOG="${root}/docker.log"
    export DUCKDB_CALL_LOG="${root}/duckdb.log"
    export SIRIUS_BENCH_WORK_DIR="${root}/work"
    export SIRIUS_BENCH_S3_BUCKET="fixture-idempotent"
    export SIRIUS_TEST_S3_BUCKET="fixture-idempotent"
    export SIRIUS_BENCH_S3_KEY="tpch/lineitem_sf10.parquet"
    export SIRIUS_TEST_S3_ENDPOINT="http://127.0.0.1:9000"
    export SIRIUS_TEST_S3_HTTPS_ENDPOINT="https://127.0.0.1:9443"
    env "$@" "${FIXTURES_SH}" --perf
  ) >"${log}" 2>&1
}

perf_parquet() {
  printf '%s/work/lineitem_sf10.parquet' "$1"
}

run_case() {
  local name="$1"
  shift
  echo "running ${name}"
  if "$@"; then
    echo "ok - ${name}"
  else
    echo "FAILED - ${name}" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

case_reuse_skips_generation() {
  local root
  root="$(mktemp -d)"
  make_stubs "${root}" fail-if-called
  local parquet
  parquet="$(perf_parquet "${root}")"
  make_parquet "${parquet}" "cached-ac1"
  write_sidecar "${parquet}"

  run_fixtures "${root}" || {
    cat "${root}/run.log" >&2
    fail "expected --perf to succeed by reusing existing fixture"
  }

  assert_empty_or_missing "${root}/duckdb.log"
  assert_contains "${root}/run.log" "[fixtures] reusing existing perf fixture"
  assert_not_contains "${root}/run.log" "[fixtures] generating SF10 lineitem benchmark parquet with"
  assert_contains "${root}/docker.log" "cp /work/lineitem_sf10.parquet local/fixture-idempotent/tpch/lineitem_sf10.parquet"
}

case_missing_regenerates_and_writes_sidecar() {
  local root
  root="$(mktemp -d)"
  make_stubs "${root}" success
  local parquet
  parquet="$(perf_parquet "${root}")"
  assert_file_not_exists "${parquet}"

  run_fixtures "${root}" || {
    cat "${root}/run.log" >&2
    fail "expected --perf to regenerate a missing fixture"
  }

  assert_nonempty "${root}/duckdb.log"
  assert_file_exists "${parquet}"
  assert_file_exists "${parquet}.sha256"
  assert_sha_valid "${parquet}"
}

case_force_regen_overrides_reuse() {
  local root
  root="$(mktemp -d)"
  make_stubs "${root}" success
  local parquet
  parquet="$(perf_parquet "${root}")"
  make_parquet "${parquet}" "old-ac3"
  write_sidecar "${parquet}"

  run_fixtures "${root}" SIRIUS_BENCH_FORCE_REGEN=1 || {
    cat "${root}/run.log" >&2
    fail "expected force regen to succeed"
  }

  assert_nonempty "${root}/duckdb.log"
  assert_contains "${root}/run.log" "[fixtures] generating SF10 lineitem benchmark parquet with"
  assert_file_exists "${parquet}.sha256"
  assert_sha_valid "${parquet}"
}

case_corrupt_fixture_is_not_uploaded() {
  local root
  root="$(mktemp -d)"
  make_stubs "${root}" fail
  local parquet
  parquet="$(perf_parquet "${root}")"
  make_parquet "${parquet}" "good-ac4"
  write_sidecar "${parquet}"
  printf 'BAD' > "${parquet}"

  if run_fixtures "${root}"; then
    cat "${root}/run.log" >&2
    fail "expected corrupt fixture with unavailable generation to fail"
  fi

  assert_nonempty "${root}/duckdb.log"
  assert_not_contains "${root}/docker.log" "cp /work/lineitem_sf10.parquet local/fixture-idempotent/tpch/lineitem_sf10.parquet"
  assert_contains "${root}/run.log" "invalid"
}

case_two_runs_are_idempotent() {
  local root
  root="$(mktemp -d)"
  make_stubs "${root}" fail-if-called
  local parquet
  parquet="$(perf_parquet "${root}")"
  make_parquet "${parquet}" "cached-ac5"
  write_sidecar "${parquet}"

  run_fixtures "${root}" || {
    cat "${root}/run.log" >&2
    fail "expected first --perf run to reuse existing fixture"
  }
  run_fixtures "${root}" || {
    cat "${root}/run.log" >&2
    fail "expected second --perf run to reuse existing fixture"
  }

  assert_empty_or_missing "${root}/duckdb.log"
  assert_sha_valid "${parquet}"
}

run_case "AC1 reuse skips generation" case_reuse_skips_generation
run_case "AC2 missing fixture regenerates and writes sidecar" case_missing_regenerates_and_writes_sidecar
run_case "AC3 force regen overrides reuse" case_force_regen_overrides_reuse
run_case "AC4 corrupt fixture is not uploaded" case_corrupt_fixture_is_not_uploaded
run_case "AC5 repeated runs are idempotent" case_two_runs_are_idempotent

if [[ "${FAILURES}" -ne 0 ]]; then
  echo "${FAILURES} fixture idempotency case(s) failed" >&2
  exit 1
fi

echo "all fixture idempotency cases passed"
