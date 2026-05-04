from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two checksum manifests written by write_artifact_manifest.py."
    )
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = compare_manifests(
        json.loads(args.expected.read_text(encoding="utf-8")),
        json.loads(args.actual.read_text(encoding="utf-8")),
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["passed"]:
        print("Artifact manifests match.")
        return
    for failure in report["failures"]:
        print(failure)
    raise SystemExit("Artifact manifest comparison failed.")


def compare_manifests(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_files = _file_map(expected)
    actual_files = _file_map(actual)
    expected_paths = set(expected_files)
    actual_paths = set(actual_files)
    failures = []
    for path in sorted(expected_paths - actual_paths):
        failures.append(f"missing_local_file:{path}")
    for path in sorted(actual_paths - expected_paths):
        failures.append(f"unexpected_local_file:{path}")
    for path in sorted(expected_paths & actual_paths):
        expected_row = expected_files[path]
        actual_row = actual_files[path]
        if expected_row.get("bytes") != actual_row.get("bytes"):
            failures.append(f"byte_mismatch:{path}")
        if expected_row.get("sha256") != actual_row.get("sha256"):
            failures.append(f"sha256_mismatch:{path}")
    if expected.get("missing_paths"):
        failures.extend(f"missing_remote_path:{path}" for path in expected["missing_paths"])
    if actual.get("missing_paths"):
        failures.extend(f"missing_local_path:{path}" for path in actual["missing_paths"])
    return {
        "schema_version": 1,
        "passed": not failures,
        "expected_file_count": len(expected_files),
        "actual_file_count": len(actual_files),
        "failures": failures,
    }


def _file_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["path"]): row
        for row in manifest.get("files", [])
        if isinstance(row, dict) and "path" in row
    }


if __name__ == "__main__":
    main()
