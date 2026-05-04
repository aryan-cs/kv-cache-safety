from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256

DEFAULT_REQUIRED_PATHS = [
    "results/h200_qwen_full_sweep",
    "results/h200_causal_patch_qwen7b",
]
REQUIRED_RAW_RUN_FILES = [
    "config.resolved.yaml",
    "environment.json",
    "manifest.json",
    "prompts.jsonl",
    "generations.jsonl",
    "cache_stats.parquet",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that local H200 evidence came from a passing checksum fetch."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--remote-manifest",
        type=Path,
        default=Path("logs/h200/h200_artifact_manifest_remote.json"),
    )
    parser.add_argument(
        "--local-manifest",
        type=Path,
        default=Path("logs/h200/h200_artifact_manifest_local.json"),
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        default=Path("logs/h200/h200_artifact_manifest_compare.json"),
    )
    parser.add_argument("--required-path", action="append", default=[])
    args = parser.parse_args()

    required_paths = args.required_path or DEFAULT_REQUIRED_PATHS
    report = check_h200_fetch_manifest(
        root=args.root,
        remote_manifest_path=args.remote_manifest,
        local_manifest_path=args.local_manifest,
        compare_report_path=args.compare_report,
        required_paths=required_paths,
    )
    if report["passed"]:
        print("H200 fetch manifest check passed.")
        return
    for failure in report["failures"]:
        print(failure)
    raise SystemExit("H200 fetch manifest check failed.")


def check_h200_fetch_manifest(
    *,
    root: Path,
    remote_manifest_path: Path,
    local_manifest_path: Path,
    compare_report_path: Path,
    required_paths: list[str],
) -> dict[str, Any]:
    root = root.resolve()
    failures: list[str] = []
    local_manifest = _read_json(local_manifest_path, failures, label="local_manifest")
    compare_report = _read_json(compare_report_path, failures, label="compare_report")
    if not local_manifest or not compare_report:
        return {"schema_version": 1, "passed": False, "failures": failures}

    if compare_report.get("passed") is not True:
        failures.append("artifact_manifest_compare_not_passed")
        for failure in compare_report.get("failures") or []:
            failures.append(f"artifact_manifest_compare_failure:{failure}")
    failures.extend(
        _compare_report_manifest_hash_failures(
            compare_report,
            remote_manifest_path=remote_manifest_path,
            local_manifest_path=local_manifest_path,
        )
    )

    requested_paths = {str(path) for path in local_manifest.get("requested_paths") or []}
    missing_paths = {str(path) for path in local_manifest.get("missing_paths") or []}
    files = {
        str(row.get("path")): row
        for row in local_manifest.get("files") or []
        if isinstance(row, dict) and row.get("path")
    }
    for required_path in required_paths:
        if required_path not in requested_paths:
            failures.append(f"missing_requested_path:{required_path}")
        if required_path in missing_paths:
            failures.append(f"manifest_marks_required_path_missing:{required_path}")
        failures.extend(_raw_file_failures(root, files, required_path))

    return {
        "schema_version": 1,
        "passed": not failures,
        "required_paths": required_paths,
        "failures": failures,
    }


def _read_json(path: Path, failures: list[str], *, label: str) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"missing_{label}:{path}")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"invalid_{label}:{path}:{exc.msg}")
        return {}
    if not isinstance(loaded, dict):
        failures.append(f"invalid_{label}:{path}:non_object")
        return {}
    return loaded


def _raw_file_failures(root: Path, files: dict[str, Any], required_path: str) -> list[str]:
    failures = []
    for filename in REQUIRED_RAW_RUN_FILES:
        rel_path = f"{required_path}/{filename}"
        row = files.get(rel_path)
        if not isinstance(row, dict):
            failures.append(f"manifest_lacks_required_raw_file:{rel_path}")
            continue
        path = root / rel_path
        if not path.exists():
            failures.append(f"missing_required_raw_file:{rel_path}")
            continue
        if row.get("bytes") != path.stat().st_size:
            failures.append(f"required_raw_file_byte_mismatch:{rel_path}")
        if row.get("sha256") != file_sha256(path):
            failures.append(f"required_raw_file_sha256_mismatch:{rel_path}")
    return failures


def _compare_report_manifest_hash_failures(
    compare_report: dict[str, Any],
    *,
    remote_manifest_path: Path,
    local_manifest_path: Path,
) -> list[str]:
    failures = []
    expected_remote_sha = compare_report.get("expected_manifest_sha256")
    if not expected_remote_sha:
        failures.append("artifact_manifest_compare_lacks_expected_manifest_sha256")
    elif not remote_manifest_path.exists():
        failures.append(f"missing_remote_manifest_copy:{remote_manifest_path}")
    elif expected_remote_sha != file_sha256(remote_manifest_path):
        failures.append("artifact_manifest_compare_expected_manifest_sha256_stale")

    expected_local_sha = compare_report.get("actual_manifest_sha256")
    if not expected_local_sha:
        failures.append("artifact_manifest_compare_lacks_actual_manifest_sha256")
    elif not local_manifest_path.exists():
        failures.append(f"missing_local_manifest:{local_manifest_path}")
    elif expected_local_sha != file_sha256(local_manifest_path):
        failures.append("artifact_manifest_compare_actual_manifest_sha256_stale")
    return failures


if __name__ == "__main__":
    main()
