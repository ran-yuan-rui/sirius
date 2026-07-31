#!/usr/bin/env python3

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "s3_test_gate_manifest.json"
DEFAULT_BINARY = Path.cwd() / "build/release/extension/sirius/test/cpp/sirius_unittest"


def fail(message):
    print(f"s3-test manifest mismatch: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_manifest(path):
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("schema_version") != 1:
        fail(f"unsupported schema version in {path}")

    names = manifest.get("test_names", [])
    expected_cases = manifest.get("expected", {}).get("test_cases")
    if len(names) != expected_cases:
        fail(
            f"manifest contains {len(names)} names but declares "
            f"{expected_cases} test cases"
        )
    if len(set(names)) != len(names):
        fail("manifest contains duplicate test names")
    return manifest


def list_selected_tests(binary, selector):
    completed = subprocess.run(
        [str(binary), selector, "--list-test-names-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    # Catch2 v2 returns the number of matched tests from list commands.
    valid_codes = {0, len(names) % 256}
    if completed.returncode not in valid_codes:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"test listing exited {completed.returncode}: {detail}")
    if not names:
        fail(f"selector {selector!r} selected no tests")
    return names


def compare_names(expected, actual):
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    if expected_counts == actual_counts:
        return

    missing = list((expected_counts - actual_counts).elements())
    added = list((actual_counts - expected_counts).elements())
    lines = []
    if missing:
        lines.append("missing:\n  " + "\n  ".join(sorted(missing)))
    if added:
        lines.append("unexpected:\n  " + "\n  ".join(sorted(added)))
    fail("\n".join(lines))


def check_run_summary(log_path, expected):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"All tests passed \((\d+) assertions in (\d+) test cases\)", text
    )
    if not matches:
        fail(f"{log_path} has no successful Catch2 summary")

    assertions, cases = (int(value) for value in matches[-1])
    expected_assertions = expected["assertions"]
    expected_cases = expected["test_cases"]
    if (assertions, cases) != (expected_assertions, expected_cases):
        fail(
            f"run reported {cases} cases / {assertions} assertions; expected "
            f"{expected_cases} / {expected_assertions}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Verify the standard S3 gate against its frozen baseline."
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--run-log",
        type=Path,
        help="optional make s3-test log whose Catch2 summary must match",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    actual_names = list_selected_tests(args.binary, manifest["selector"])
    compare_names(manifest["test_names"], actual_names)
    if args.run_log is not None:
        check_run_summary(args.run_log, manifest["expected"])

    baseline = manifest["baseline"]
    checked = f"{len(actual_names)} test names"
    if args.run_log is not None:
        checked += f" / {manifest['expected']['assertions']} assertions"
    print(
        f"s3-test manifest matches: {checked} "
        f"(baseline Sirius {baseline['sirius_sha']}, "
        f"cuCascade {baseline['cucascade_sha']})"
    )


if __name__ == "__main__":
    main()
