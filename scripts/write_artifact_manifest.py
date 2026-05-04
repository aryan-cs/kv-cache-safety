from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, utc_timestamp, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a checksum manifest for result and paper artifact trees."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        required=True,
        help="File or directory path relative to --root. Repeat for multiple paths.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = artifact_manifest(args.root, [Path(path) for path in args.paths])
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")


def artifact_manifest(root: Path, paths: list[Path]) -> dict[str, Any]:
    root = root.resolve()
    files = []
    missing = []
    for raw_path in paths:
        if raw_path.is_absolute() or ".." in raw_path.parts:
            raise SystemExit(f"Artifact path must be relative and inside the repo: {raw_path}")
        path = root / raw_path
        if not path.exists():
            missing.append(raw_path.as_posix())
            continue
        for file_path in _iter_files(path):
            rel_path = file_path.resolve().relative_to(root).as_posix()
            files.append(
                {
                    "path": rel_path,
                    "bytes": file_path.stat().st_size,
                    "sha256": file_sha256(file_path),
                }
            )
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "root": str(root),
        "requested_paths": [path.as_posix() for path in paths],
        "missing_paths": missing,
        "file_count": len(files),
        "files": sorted(files, key=lambda row: row["path"]),
    }


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(child for child in path.rglob("*") if child.is_file())


if __name__ == "__main__":
    main()
