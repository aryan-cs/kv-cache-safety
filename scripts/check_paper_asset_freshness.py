from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from export_paper_assets import TABLE_FILES as EXPORTED_TABLE_FILES
from export_paper_assets import export_paper_assets

from cache_safety_erasure.utils.io import file_sha256

SOURCE_ARTIFACTS = ["manifest.json", "metrics.json", "figures/manifest.json"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if generated paper tables/macros are stale relative to result artifacts."
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Paper/results pair in the form paper/generated/run=results/run.",
    )
    parser.add_argument(
        "--required-table",
        action="append",
        default=[],
        help="Generated paper table/macro filename that must be manifest-pinned.",
    )
    parser.add_argument(
        "--require-exported-table-set",
        action="store_true",
        help="Require the full table/macro set written by export_paper_assets.py.",
    )
    parser.add_argument(
        "--require-recomputed-output",
        action="store_true",
        help="Re-run export_paper_assets.py into a temp dir and compare generated outputs.",
    )
    parser.add_argument(
        "--macro-prefix",
        default=None,
        help="Override the inferred LaTeX macro prefix used for recomputed output.",
    )
    args = parser.parse_args()

    required_tables = list(args.required_table)
    if args.require_exported_table_set:
        required_tables.extend(EXPORTED_TABLE_FILES)
    required_tables = sorted(set(required_tables))
    failures: list[str] = []
    for value in args.pair:
        paper_dir, results_dir = _parse_pair(value)
        failures.extend(
            check_paper_asset_freshness(
                paper_dir,
                results_dir,
                required_tables=required_tables or None,
                require_recomputed_output=args.require_recomputed_output,
                macro_prefix=args.macro_prefix,
            )
        )
    if failures:
        print("PAPER ASSET FRESHNESS CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PAPER ASSET FRESHNESS CHECK PASSED")


def check_paper_asset_freshness(
    paper_dir: Path,
    results_dir: Path,
    *,
    required_tables: list[str] | None = None,
    require_recomputed_output: bool = False,
    macro_prefix: str | None = None,
) -> list[str]:
    failures: list[str] = []
    manifest_path = paper_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return [f"missing paper artifact manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid paper artifact manifest `{manifest_path}`: {exc}"]
    failures.extend(_table_failures(manifest, paper_dir, required_tables=required_tables))
    failures.extend(_source_failures(manifest, results_dir))
    failures.extend(_provenance_failures(manifest, results_dir))
    if require_recomputed_output:
        failures.extend(
            _recomputed_output_failures(
                manifest,
                paper_dir,
                results_dir,
                required_tables=required_tables,
                macro_prefix=macro_prefix,
            )
        )
    return failures


def _recomputed_output_failures(
    manifest: dict[str, Any],
    paper_dir: Path,
    results_dir: Path,
    *,
    required_tables: list[str] | None,
    macro_prefix: str | None,
) -> list[str]:
    failures: list[str] = []
    expected_files = sorted(set(required_tables or EXPORTED_TABLE_FILES))
    with tempfile.TemporaryDirectory(prefix="paper_asset_recompute_") as tmp:
        recomputed_dir = Path(tmp) / "generated"
        try:
            export_paper_assets(
                results_dir,
                recomputed_dir,
                _resolve_macro_prefix(manifest, paper_dir, results_dir, macro_prefix),
            )
        except (Exception, SystemExit) as exc:
            return [f"paper artifact recompute failed for {results_dir}: {exc}"]
        for name in expected_files:
            current_path = paper_dir / name
            recomputed_path = recomputed_dir / name
            if not current_path.exists():
                failures.append(f"paper artifact generated output `{name}` is missing")
                continue
            if not recomputed_path.exists():
                failures.append(f"paper artifact recompute did not write `{name}`")
                continue
            if current_path.read_bytes() != recomputed_path.read_bytes():
                failures.append(
                    f"paper artifact generated output `{name}` differs from metrics export"
                )
    return failures


def _resolve_macro_prefix(
    manifest: dict[str, Any],
    paper_dir: Path,
    results_dir: Path,
    override: str | None,
) -> str:
    if override:
        return override
    manifest_prefix = manifest.get("macro_prefix")
    if isinstance(manifest_prefix, str) and manifest_prefix.strip():
        return manifest_prefix
    macros_path = paper_dir / "result_macros.tex"
    try:
        macros_text = macros_path.read_text(encoding="utf-8")
    except OSError:
        return _infer_macro_prefix(paper_dir, results_dir)
    match = re.search(r"\\renewcommand\{\\([A-Za-z]+)RunId\}", macros_text)
    if match:
        return match.group(1)
    return _infer_macro_prefix(paper_dir, results_dir)


def _infer_macro_prefix(paper_dir: Path, results_dir: Path) -> str:
    label = f"{paper_dir.name} {results_dir.name}".lower()
    if "causal" in label:
        return "Causal"
    if "qwen32" in label or "32b" in label or "thirty" in label:
        return "QwenThirtyTwo"
    return "Primary"


def _table_failures(
    manifest: dict[str, Any],
    paper_dir: Path,
    *,
    required_tables: list[str] | None = None,
) -> list[str]:
    failures: list[str] = []
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or not tables:
        return [f"paper artifact manifest lacks table entries: {paper_dir}"]
    for name in required_tables or []:
        if name not in tables:
            failures.append(f"paper artifact manifest lacks required table `{name}`")
    for name, table in tables.items():
        if not isinstance(table, dict):
            failures.append(f"paper artifact table entry `{name}` is malformed")
            continue
        path = Path(str(table.get("path", "")))
        expected_path = paper_dir / name
        if name in (required_tables or []) and path.resolve() != expected_path.resolve():
            failures.append(f"paper artifact required table `{name}` path is unexpected")
        try:
            path.resolve().relative_to(paper_dir.resolve())
        except ValueError:
            failures.append(f"paper artifact table `{name}` path is outside paper dir")
        if not path.exists():
            failures.append(f"paper artifact table `{name}` is missing")
            continue
        if table.get("sha256") != file_sha256(path):
            failures.append(f"paper artifact table `{name}` hash is stale")
        if "bytes" in table and table.get("bytes") != path.stat().st_size:
            failures.append(f"paper artifact table `{name}` byte count is stale")
    return failures


def _source_failures(manifest: dict[str, Any], results_dir: Path) -> list[str]:
    failures: list[str] = []
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, dict):
        return [f"paper artifact manifest lacks source entries for {results_dir}"]
    for name in SOURCE_ARTIFACTS:
        source = sources.get(name)
        if not isinstance(source, dict):
            failures.append(f"paper artifact manifest lacks source `{name}`")
            continue
        path = results_dir / name
        if not path.exists():
            failures.append(f"paper artifact source `{name}` is missing")
            continue
        if source.get("sha256") != file_sha256(path):
            failures.append(f"paper artifact source `{name}` hash is stale")
    return failures


def _provenance_failures(manifest: dict[str, Any], results_dir: Path) -> list[str]:
    failures = []
    if not manifest.get("analysis_git_commit"):
        failures.append(f"paper artifact manifest lacks analysis git commit: {results_dir}")
    if manifest.get("analysis_git_dirty"):
        failures.append(f"paper artifact manifest was generated from a dirty analysis tree: {results_dir}")
    if manifest.get("source_run_git_dirty"):
        failures.append(f"paper artifact manifest source run was dirty: {results_dir}")
    run_manifest_path = results_dir / "manifest.json"
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        run_manifest = {}
    expected_run_commit = run_manifest.get("git_commit") if isinstance(run_manifest, dict) else None
    observed_run_commit = manifest.get("source_run_git_commit")
    if expected_run_commit and observed_run_commit != expected_run_commit:
        failures.append("paper artifact manifest source run git commit is stale")
    elif not observed_run_commit:
        failures.append(f"paper artifact manifest lacks source run git commit: {results_dir}")
    return failures


def _parse_pair(value: str) -> tuple[Path, Path]:
    if "=" not in value:
        raise SystemExit(f"Expected --pair value like paper/generated/run=results/run, got `{value}`")
    paper_dir, results_dir = value.split("=", 1)
    return Path(paper_dir), Path(results_dir)


if __name__ == "__main__":
    main()
