#!/usr/bin/env bash
set -euo pipefail

# Manual real-AWS sweep. P is the total slot count, R is the reactor count, and
# C=P/R is passed as max_connections. Export temporary SIRIUS_TEST_S3_* credentials,
# then set SWEEP_MODE, SWEEP_P, SWEEP_R, and RAW_S3_KEY as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIRIUS_PROJECT_ROOT="${SIRIUS_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
S3_TEST_BIN="${S3_TEST_BIN:-${SIRIUS_PROJECT_ROOT}/build/release/extension/sirius/test/cpp/sirius_unittest}"
SIRIUS_UNITTEST_LOG_DIR="${SIRIUS_UNITTEST_LOG_DIR:-${SIRIUS_PROJECT_ROOT}/build/release/extension/sirius/test/cpp/log}"
SWEEP_OUTPUT_PATH="${SWEEP_OUTPUT_PATH:-${SIRIUS_PROJECT_ROOT}/doc/s3support/perf-history-concurrency.jsonl}"
SWEEP_ARTIFACT_DIR="${SWEEP_ARTIFACT_DIR:-${SIRIUS_UNITTEST_LOG_DIR}/concurrency-sweep}"

SWEEP_MODE="${SWEEP_MODE:-both}"
SWEEP_P="${SWEEP_P:-1,2,4,8,16,32,64,128}"
SWEEP_R="${SWEEP_R:-1,2,4}"
SWEEP_MIN_SECONDS="${SWEEP_MIN_SECONDS:-30}"
SWEEP_IDLE_SECONDS="${SWEEP_IDLE_SECONDS:-3}"
SWEEP_PINNED_BYTES_PER_SLOT="${SWEEP_PINNED_BYTES_PER_SLOT:-1048576}"
SWEEP_PINNED_BUDGET_BYTES="${SWEEP_PINNED_BUDGET_BYTES:-8589934592}"
RAW_RANGE_BYTES="${RAW_RANGE_BYTES:-1048576}"
RAW_REPS="${RAW_REPS:-1}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

fail()
{
  echo "run_concurrency_sweep: $*" >&2
  exit 1
}

require_env()
{
  local name=$1
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

parse_positive_csv()
{
  local raw=$1
  local output_name=$2
  local -n output=$output_name
  local token
  local -a tokens

  [[ -n "$raw" && "$raw" != ,* && "$raw" != *, && "$raw" != *,,* ]] ||
    fail "invalid positive-integer list: ${raw}"
  IFS=',' read -r -a tokens <<<"$raw"
  output=()
  for token in "${tokens[@]}"; do
    if [[ "$token" =~ ^[[:space:]]*([1-9][0-9]*)[[:space:]]*$ ]]; then
      output+=("${BASH_REMATCH[1]}")
    else
      fail "invalid positive integer '${token}' in '${raw}'"
    fi
  done
}

positive_integer()
{
  local name=$1
  local value=$2
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer"
}

nonnegative_integer()
{
  local name=$1
  local value=$2
  [[ "$value" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer"
}

case "$SWEEP_MODE" in
  raw | sql | both) ;;
  *) fail "SWEEP_MODE must be raw, sql, or both" ;;
esac

[[ -x "$S3_TEST_BIN" ]] || fail "${S3_TEST_BIN} not found; run make release first"
command -v python3 >/dev/null || fail "python3 is required to merge benchmark JSON"
command -v ip >/dev/null || fail "ip is required to discover the benchmark interface"

for name in SIRIUS_TEST_S3_ENDPOINT SIRIUS_TEST_S3_REGION SIRIUS_TEST_S3_ACCESS_KEY \
  SIRIUS_TEST_S3_SECRET_KEY SIRIUS_TEST_S3_SESSION_TOKEN SIRIUS_TEST_S3_BUCKET; do
  require_env "$name"
done
if [[ "$SWEEP_MODE" == raw || "$SWEEP_MODE" == both ]]; then
  require_env RAW_S3_KEY
fi

positive_integer SWEEP_IDLE_SECONDS "$SWEEP_IDLE_SECONDS"
positive_integer SWEEP_PINNED_BYTES_PER_SLOT "$SWEEP_PINNED_BYTES_PER_SLOT"
positive_integer SWEEP_PINNED_BUDGET_BYTES "$SWEEP_PINNED_BUDGET_BYTES"
positive_integer RAW_RANGE_BYTES "$RAW_RANGE_BYTES"
positive_integer RAW_REPS "$RAW_REPS"
nonnegative_integer SWEEP_MIN_SECONDS "$SWEEP_MIN_SECONDS"

declare -a P_VALUES R_VALUES
parse_positive_csv "$SWEEP_P" P_VALUES
parse_positive_csv "$SWEEP_R" R_VALUES

IFACE="${SWEEP_IFACE:-$(ip -o -4 route show to default | awk 'NR == 1 {print $5}')}"
[[ -n "$IFACE" ]] || fail "could not discover the default network interface"
grep -qE "^[[:space:]]*${IFACE}:" /proc/net/dev ||
  fail "${IFACE} is not present in /proc/net/dev"

if [[ "${SWEEP_ALLOW_BUSY_HOST:-0}" != 1 ]] && pgrep -x sirius_unittest >/dev/null 2>&1; then
  fail "another sirius_unittest is running; use an idle host or set SWEEP_ALLOW_BUSY_HOST=1"
fi

mkdir -p "$SIRIUS_UNITTEST_LOG_DIR" "$SWEEP_ARTIFACT_DIR" "$(dirname "$SWEEP_OUTPUT_PATH")"

export SIRIUS_PROJECT_ROOT SIRIUS_UNITTEST_LOG_DIR
export SIRIUS_TEST_S3_STRICT=1
export SIRIUS_BENCH_GIT_SHA="${SIRIUS_BENCH_GIT_SHA:-$(git -C "$SIRIUS_PROJECT_ROOT" rev-parse --short HEAD)}"
export HOSTNAME="${HOSTNAME:-$(hostname)}"
if [[ -z "${SIRIUS_BENCH_S3_KEY:-}" && -n "${RAW_S3_KEY:-}" ]]; then
  export SIRIUS_BENCH_S3_KEY="$RAW_S3_KEY"
fi

rx_bytes()
{
  awk -F: -v iface="$IFACE" '
    {
      name=$1
      gsub(/[[:space:]]/, "", name)
      if (name == iface) {
        count=split($2, fields, /[[:space:]]+/)
        for (i = 1; i <= count; ++i) {
          if (fields[i] != "") {
            print fields[i]
            exit
          }
        }
      }
    }' /proc/net/dev
}

ena_counter()
{
  local counter=$1
  local value
  if ! command -v ethtool >/dev/null; then
    echo null
    return
  fi
  if ! value=$(ethtool -S "$IFACE" 2>/dev/null | awk -F: -v wanted="$counter" '
    {
      name=$1
      gsub(/[[:space:]]/, "", name)
      if (name == wanted) {
        value=$2
        gsub(/[[:space:]]/, "", value)
        print value
        exit
      }
    }'); then
    echo null
    return
  fi
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
  else
    echo null
  fi
}

socket_count()
{
  local port=$1
  if ! command -v ss >/dev/null; then
    echo null
    return
  fi
  local value
  if ! value=$(ss -tanH 2>/dev/null |
    awk -v suffix=":${port}" '$1 == "ESTAB" && substr($5, length($5) - length(suffix) + 1) == suffix {
        count++
      } END {print count + 0}'); then
    echo null
    return
  fi
  echo "$value"
}

MONITOR_PIDS=()
SAR_DEV_LOG=""
MPSTAT_LOG=""
GPU_LOG=""
SOCKET_LOG=""

start_monitors()
{
  local label=$1
  MONITOR_PIDS=()
  SAR_DEV_LOG=""
  MPSTAT_LOG=""
  GPU_LOG=""
  SOCKET_LOG="${SWEEP_ARTIFACT_DIR}/sockets_${label}.log"

  if command -v sar >/dev/null; then
    SAR_DEV_LOG="${SWEEP_ARTIFACT_DIR}/sar_dev_${label}.log"
    sar -n DEV 1 >"$SAR_DEV_LOG" 2>&1 &
    MONITOR_PIDS+=("$!")
  fi
  if command -v mpstat >/dev/null; then
    MPSTAT_LOG="${SWEEP_ARTIFACT_DIR}/mpstat_${label}.log"
    mpstat -P ALL 1 >"$MPSTAT_LOG" 2>&1 &
    MONITOR_PIDS+=("$!")
  fi
  if command -v nvidia-smi >/dev/null; then
    GPU_LOG="${SWEEP_ARTIFACT_DIR}/gpu_${label}.log"
    nvidia-smi dmon -s u -d 1 >"$GPU_LOG" 2>&1 &
    MONITOR_PIDS+=("$!")
  fi
  (
    while :; do
      printf '%s %s %s\n' "$(date +%s.%N)" "$(socket_count 80)" "$(socket_count 443)"
      sleep 0.5
    done
  ) >"$SOCKET_LOG" 2>&1 &
  MONITOR_PIDS+=("$!")
}

stop_monitors()
{
  local pid
  for pid in "${MONITOR_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${MONITOR_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  MONITOR_PIDS=()
}

trap stop_monitors EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

socket_peak()
{
  local column=$1
  if [[ ! -s "$SOCKET_LOG" ]]; then
    echo null
    return
  fi
  awk -v column="$column" '
    $column ~ /^[0-9]+$/ && $column > peak {peak=$column; found=1}
    END {if (found) print peak; else print "null"}' "$SOCKET_LOG"
}

newest_json_after()
{
  local marker=$1
  local mode=$2
  local file
  local newest=""
  local -a candidates

  shopt -s nullglob
  if [[ "$mode" == raw ]]; then
    candidates=("${SIRIUS_UNITTEST_LOG_DIR}"/s3_rest_perf_raw_transport_*.json)
  else
    candidates=("${SIRIUS_UNITTEST_LOG_DIR}"/s3_rest_perf_*.json)
  fi
  shopt -u nullglob
  for file in "${candidates[@]}"; do
    if [[ "$mode" == sql && "$file" == *s3_rest_perf_raw_transport_* ]]; then
      continue
    fi
    if [[ "$file" -nt "$marker" && ( -z "$newest" || "$file" -nt "$newest" ) ]]; then
      newest=$file
    fi
  done
  printf '%s' "$newest"
}

emit_record()
{
  local bench_json=$1
  python3 - "$bench_json" "$SWEEP_OUTPUT_PATH" <<'PY'
import datetime
import json
import os
import pathlib
import sys


def integer(name):
    value = os.environ.get(name, "")
    return int(value) if value not in ("", "null") else None


def delta(after_name, before_name):
    after = integer(after_name)
    before = integer(before_name)
    if after is None or before is None or after < before:
        return None
    return after - before


bench_path = sys.argv[1]
history_path = pathlib.Path(sys.argv[2])
bench = None
if bench_path:
    with open(bench_path, encoding="utf-8") as source:
        bench = json.load(source)

elapsed_ns = integer("POINT_ELAPSED_NS")
elapsed_s = elapsed_ns / 1_000_000_000 if elapsed_ns is not None else None
nic_bytes = delta("POINT_RX_AFTER", "POINT_RX_BEFORE")
idle_bps = float(os.environ["POINT_IDLE_RX_BPS"])
idle_adjusted = None
if nic_bytes is not None and elapsed_s is not None:
    idle_adjusted = max(0, round(nic_bytes - idle_bps * elapsed_s))

def gbps(byte_count):
    if byte_count is None or not elapsed_s:
        return None
    return byte_count * 8 / elapsed_s / 1_000_000_000


record = {
    "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_sha": os.environ["SIRIUS_BENCH_GIT_SHA"],
    "hostname": os.environ["HOSTNAME"],
    "mode": os.environ["POINT_MODE"],
    "status": os.environ["POINT_STATUS"],
    "exit_code": integer("POINT_EXIT_CODE"),
    "skip_reason": os.environ.get("POINT_SKIP_REASON") or None,
    "P": integer("POINT_P"),
    "R": integer("POINT_R"),
    "C": integer("POINT_C"),
    "parameters": {
        "raw_key": os.environ.get("RAW_S3_KEY") or None,
        "sql_key": os.environ.get("SIRIUS_BENCH_S3_KEY") or None,
        "raw_range_bytes": integer("RAW_RANGE_BYTES"),
        "raw_min_seconds": integer("SWEEP_MIN_SECONDS"),
        "raw_reps": integer("RAW_REPS"),
        "pinned_bytes_per_slot": integer("SWEEP_PINNED_BYTES_PER_SLOT"),
        "pinned_required_bytes": integer("POINT_PINNED_REQUIRED_BYTES"),
        "pinned_budget_bytes": integer("SWEEP_PINNED_BUDGET_BYTES"),
    },
    "host_metrics": {
        "interface": os.environ["POINT_IFACE"],
        "elapsed_s": elapsed_s,
        "idle_rx_bytes_per_s": idle_bps,
        "nic_rx_bytes": nic_bytes,
        "nic_rx_idle_adjusted_bytes": idle_adjusted,
        "nic_rx_gbps": gbps(nic_bytes),
        "nic_rx_idle_adjusted_gbps": gbps(idle_adjusted),
        "ena_allowance_exceeded_delta": {
            "bw_in": delta("POINT_ENA_BW_IN_AFTER", "POINT_ENA_BW_IN_BEFORE"),
            "bw_out": delta("POINT_ENA_BW_OUT_AFTER", "POINT_ENA_BW_OUT_BEFORE"),
            "pps": delta("POINT_ENA_PPS_AFTER", "POINT_ENA_PPS_BEFORE"),
        },
        "established_sockets": {
            "port_80_before": integer("POINT_SOCKET_80_BEFORE"),
            "port_80_after": integer("POINT_SOCKET_80_AFTER"),
            "port_80_peak": integer("POINT_SOCKET_80_PEAK"),
            "port_443_before": integer("POINT_SOCKET_443_BEFORE"),
            "port_443_after": integer("POINT_SOCKET_443_AFTER"),
            "port_443_peak": integer("POINT_SOCKET_443_PEAK"),
        },
        "sampler_logs": {
            "sar_dev": os.environ.get("POINT_SAR_DEV_LOG") or None,
            "mpstat": os.environ.get("POINT_MPSTAT_LOG") or None,
            "nvidia_smi_dmon": os.environ.get("POINT_GPU_LOG") or None,
            "sockets": os.environ.get("POINT_SOCKET_LOG") or None,
            "test": os.environ.get("POINT_TEST_LOG") or None,
        },
    },
    "bench": bench,
}

line = json.dumps(record, separators=(",", ":"), sort_keys=True)
history_path.parent.mkdir(parents=True, exist_ok=True)
with history_path.open("a", encoding="utf-8") as output:
    output.write(line + "\n")
print(line)
PY
}

measure_idle()
{
  local before after
  before=$(rx_bytes)
  sleep "$SWEEP_IDLE_SECONDS"
  after=$(rx_bytes)
  awk -v before="$before" -v after="$after" -v seconds="$SWEEP_IDLE_SECONDS" \
    'BEGIN {delta=after-before; if (delta < 0) delta=0; printf "%.3f", delta/seconds}'
}

run_cell()
{
  local mode=$1
  local p=$2
  local r=$3
  local c=$4
  local label="${mode}_P${p}_R${r}C${c}"
  local marker="${SWEEP_ARTIFACT_DIR}/marker_${label}"
  local test_log="${SWEEP_ARTIFACT_DIR}/test_${label}.log"
  local pinned_required
  local rx_before rx_after start_ns stop_ns rc bench_json
  local ena_bw_in_before ena_bw_in_after ena_bw_out_before ena_bw_out_after
  local ena_pps_before ena_pps_after socket_80_before socket_80_after socket_443_before
  local socket_443_after socket_80_peak socket_443_peak

  if ((p > 9223372036854775807 / SWEEP_PINNED_BYTES_PER_SLOT)); then
    fail "P * SWEEP_PINNED_BYTES_PER_SLOT overflows signed 64-bit arithmetic"
  fi
  pinned_required=$((p * SWEEP_PINNED_BYTES_PER_SLOT))

  export POINT_MODE="$mode" POINT_P="$p" POINT_R="$r" POINT_C="$c"
  export POINT_IFACE="$IFACE" POINT_PINNED_REQUIRED_BYTES="$pinned_required"
  export POINT_TEST_LOG="$test_log"
  export POINT_RX_BEFORE=null POINT_RX_AFTER=null POINT_ELAPSED_NS=null
  export POINT_ENA_BW_IN_BEFORE=null POINT_ENA_BW_IN_AFTER=null
  export POINT_ENA_BW_OUT_BEFORE=null POINT_ENA_BW_OUT_AFTER=null
  export POINT_ENA_PPS_BEFORE=null POINT_ENA_PPS_AFTER=null
  export POINT_SOCKET_80_BEFORE=null POINT_SOCKET_80_AFTER=null POINT_SOCKET_80_PEAK=null
  export POINT_SOCKET_443_BEFORE=null POINT_SOCKET_443_AFTER=null POINT_SOCKET_443_PEAK=null
  export POINT_SAR_DEV_LOG="" POINT_MPSTAT_LOG="" POINT_GPU_LOG="" POINT_SOCKET_LOG=""
  export POINT_EXIT_CODE=0 POINT_SKIP_REASON=""

  if ((pinned_required > SWEEP_PINNED_BUDGET_BYTES)); then
    export POINT_STATUS=skipped
    export POINT_SKIP_REASON="pinned memory preflight exceeded"
    emit_record ""
    return
  fi

  : >"$marker"
  socket_80_before=$(socket_count 80)
  socket_443_before=$(socket_count 443)
  ena_bw_in_before=$(ena_counter bw_in_allowance_exceeded)
  ena_bw_out_before=$(ena_counter bw_out_allowance_exceeded)
  ena_pps_before=$(ena_counter pps_allowance_exceeded)
  rx_before=$(rx_bytes)
  start_monitors "$label"
  start_ns=$(date +%s%N)

  set +e
  if [[ "$mode" == raw ]]; then
    RAW_REST_N_REACTORS="$r" \
      RAW_MAX_CONNECTIONS="$c" \
      RAW_RANGE_BYTES="$RAW_RANGE_BYTES" \
      RAW_MIN_SECONDS="$SWEEP_MIN_SECONDS" \
      RAW_REPS="$RAW_REPS" \
      "$S3_TEST_BIN" "[s3][bench][raw-transport]" >"$test_log" 2>&1
    rc=$?
  else
    SIRIUS_BENCH_MC="$c" \
      SIRIUS_BENCH_N_REACTORS="$r" \
      "$S3_TEST_BIN" "[s3][bench][aws]~[raw-transport]" >"$test_log" 2>&1
    rc=$?
  fi
  set -e

  stop_ns=$(date +%s%N)
  stop_monitors
  rx_after=$(rx_bytes)
  ena_bw_in_after=$(ena_counter bw_in_allowance_exceeded)
  ena_bw_out_after=$(ena_counter bw_out_allowance_exceeded)
  ena_pps_after=$(ena_counter pps_allowance_exceeded)
  socket_80_after=$(socket_count 80)
  socket_443_after=$(socket_count 443)
  socket_80_peak=$(socket_peak 2)
  socket_443_peak=$(socket_peak 3)

  bench_json=""
  if ((rc == 0)); then
    bench_json=$(newest_json_after "$marker" "$mode")
    if [[ -z "$bench_json" ]]; then
      rc=2
      echo "run_concurrency_sweep: ${label} produced no new benchmark JSON" >>"$test_log"
    fi
  fi

  export POINT_RX_BEFORE="$rx_before" POINT_RX_AFTER="$rx_after"
  export POINT_ELAPSED_NS="$((stop_ns - start_ns))"
  export POINT_ENA_BW_IN_BEFORE="$ena_bw_in_before" POINT_ENA_BW_IN_AFTER="$ena_bw_in_after"
  export POINT_ENA_BW_OUT_BEFORE="$ena_bw_out_before" POINT_ENA_BW_OUT_AFTER="$ena_bw_out_after"
  export POINT_ENA_PPS_BEFORE="$ena_pps_before" POINT_ENA_PPS_AFTER="$ena_pps_after"
  export POINT_SOCKET_80_BEFORE="$socket_80_before" POINT_SOCKET_80_AFTER="$socket_80_after"
  export POINT_SOCKET_80_PEAK="$socket_80_peak"
  export POINT_SOCKET_443_BEFORE="$socket_443_before" POINT_SOCKET_443_AFTER="$socket_443_after"
  export POINT_SOCKET_443_PEAK="$socket_443_peak"
  export POINT_SAR_DEV_LOG="$SAR_DEV_LOG" POINT_MPSTAT_LOG="$MPSTAT_LOG"
  export POINT_GPU_LOG="$GPU_LOG" POINT_SOCKET_LOG="$SOCKET_LOG"
  export POINT_EXIT_CODE="$rc"
  if ((rc == 0)); then
    export POINT_STATUS=ok
  else
    export POINT_STATUS=failed
  fi
  emit_record "$bench_json"

  if ((rc != 0)); then
    echo "run_concurrency_sweep: ${label} failed; see ${test_log}" >&2
    return "$rc"
  fi
}

IDLE_RX_BPS=$(measure_idle)
export POINT_IDLE_RX_BPS="$IDLE_RX_BPS"
echo "[concurrency-sweep] mode=${SWEEP_MODE} iface=${IFACE} idle_rx_Bps=${IDLE_RX_BPS}"
echo "[concurrency-sweep] output=${SWEEP_OUTPUT_PATH}"

for r in "${R_VALUES[@]}"; do
  for p in "${P_VALUES[@]}"; do
    if ((p % r != 0)); then
      echo "[concurrency-sweep] skip P=${p}, R=${r}: P must be divisible by R" >&2
      continue
    fi
    c=$((p / r))
    if [[ "$SWEEP_MODE" == raw || "$SWEEP_MODE" == both ]]; then
      run_cell raw "$p" "$r" "$c"
    fi
    if [[ "$SWEEP_MODE" == sql || "$SWEEP_MODE" == both ]]; then
      run_cell sql "$p" "$r" "$c"
    fi
  done
done

echo "[concurrency-sweep] complete: ${SWEEP_OUTPUT_PATH}"
