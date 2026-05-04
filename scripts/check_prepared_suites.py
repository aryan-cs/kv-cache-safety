from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.evals.io import processed_suite_path
from cache_safety_erasure.utils.io import file_sha256, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prepared prompt suites before launching expensive model runs."
    )
    parser.add_argument("--suite", action="append", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-records", type=int, default=100)
    parser.add_argument(
        "--suite-min-records",
        action="append",
        default=[],
        help="Optional per-suite threshold override, e.g. system_leakage=2.",
    )
    parser.add_argument("--require-public-provenance", action="store_true")
    args = parser.parse_args()

    failures = check_prepared_suites(
        suites=args.suite,
        data_dir=args.data_dir,
        min_records=args.min_records,
        suite_min_records=parse_suite_min_records(args.suite_min_records),
        require_public_provenance=args.require_public_provenance,
    )
    if failures:
        print("PREPARED SUITE CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PREPARED SUITE CHECK PASSED")


def parse_suite_min_records(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Expected --suite-min-records value like suite=100, got `{value}`")
        suite, raw_count = value.split("=", 1)
        parsed[suite] = int(raw_count)
    return parsed


def check_prepared_suites(
    *,
    suites: list[str],
    data_dir: Path,
    min_records: int,
    suite_min_records: dict[str, int],
    require_public_provenance: bool,
) -> list[str]:
    failures: list[str] = []
    for suite in suites:
        path = processed_suite_path(suite, data_dir)
        manifest_path = path.with_suffix(".manifest.json")
        required_count = suite_min_records.get(suite, min_records)
        if not path.exists():
            failures.append(f"`{suite}` is missing prepared JSONL: {path}")
            continue
        records = read_jsonl(path)
        if len(records) < required_count:
            failures.append(f"`{suite}` has {len(records)} records; need >= {required_count}")
        _check_manifest(suite, path, manifest_path, records, failures)
        if require_public_provenance and suite.startswith("public_"):
            _check_public_provenance(suite, records, failures)
    return failures


def _check_manifest(
    suite: str,
    path: Path,
    manifest_path: Path,
    records: list[dict[str, Any]],
    failures: list[str],
) -> None:
    if not manifest_path.exists():
        failures.append(f"`{suite}` is missing suite manifest: {manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"`{suite}` has invalid suite manifest: {exc}")
        return
    if manifest.get("suite_name") != suite:
        failures.append(f"`{suite}` manifest suite_name is `{manifest.get('suite_name')}`")
    if int(manifest.get("record_count") or -1) != len(records):
        failures.append(
            f"`{suite}` manifest record_count {manifest.get('record_count')} "
            f"does not match JSONL count {len(records)}"
        )
    if manifest.get("sha256") != file_sha256(path):
        failures.append(f"`{suite}` manifest sha256 is stale")


def _check_public_provenance(
    suite: str, records: list[dict[str, Any]], failures: list[str]
) -> None:
    missing = 0
    for record in records:
        metadata = record.get("metadata") or {}
        if not metadata.get("source_dataset") or not metadata.get("source_split"):
            missing += 1
    if missing:
        failures.append(f"`{suite}` has {missing} public records without dataset provenance")


if __name__ == "__main__":
    main()
