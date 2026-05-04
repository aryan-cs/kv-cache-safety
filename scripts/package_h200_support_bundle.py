from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import utc_timestamp, write_json

DEFAULT_SUPPORT_BUNDLE = Path("logs/h200/h200_support_bundle_latest.tar.gz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Package non-experiment H200 infrastructure diagnostics for support handoff."
        )
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=Path("logs/h200/h200_status_latest.json"),
        help="JSON produced by scripts/report_h200_status.py.",
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=Path("logs/h200/h200_status_latest.md"),
    )
    parser.add_argument(
        "--admin-md",
        type=Path,
        default=Path("logs/h200/h200_admin_report.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUPPORT_BUNDLE,
        help="Output tarball path. Defaults to the stable latest-path used by support reports.",
    )
    args = parser.parse_args()

    bundle = package_support_bundle(
        status_json=args.status_json,
        status_md=args.status_md,
        admin_md=args.admin_md,
        output=args.output,
    )
    print(f"Wrote {bundle}")


def package_support_bundle(
    *,
    status_json: Path,
    status_md: Path,
    admin_md: Path,
    output: Path,
) -> Path:
    if not status_json.exists():
        raise SystemExit(
            f"Missing status JSON: {status_json}. "
            "Run scripts/report_h200_status.py with --output-json first."
        )
    status = json.loads(status_json.read_text(encoding="utf-8"))
    inputs = support_bundle_inputs(status, status_json=status_json, status_md=status_md, admin_md=admin_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix("").with_suffix(".manifest.json")
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_timestamp(),
            "purpose": "non-experiment H200 infrastructure diagnostics",
            "contains_model_generations": False,
            "contains_paper_evidence": False,
            "status_created_at_utc": status.get("created_at_utc"),
            "repo_dir": status.get("repo_dir"),
            "git_commit": (status.get("git") or {}).get("commit"),
            "launcher_waiting": status.get("launcher_waiting"),
            "gpu_gate_block_reasons": status.get("gpu_gate_block_reasons"),
            "hidden_gpu_context_likely": status.get("hidden_gpu_context_likely"),
            "files": [{"arcname": arcname, "path": str(path)} for arcname, path in inputs],
        },
    )
    with tarfile.open(output, "w:gz") as archive:
        archive.add(manifest_path, arcname="manifest.json")
        for arcname, path in inputs:
            archive.add(path, arcname=arcname)
    return output


def support_bundle_inputs(
    status: dict[str, Any],
    *,
    status_json: Path,
    status_md: Path,
    admin_md: Path,
) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for arcname, path in [
        ("h200_status_latest.json", status_json),
        ("h200_status_latest.md", status_md),
        ("h200_admin_report.md", admin_md),
    ]:
        if path.exists():
            inputs.append((arcname, path))
    launcher_log = Path(str((status.get("launcher_log") or {}).get("path") or ""))
    if launcher_log.exists():
        inputs.append(("launcher_log_tail_source.log", launcher_log))
    return inputs


if __name__ == "__main__":
    main()
