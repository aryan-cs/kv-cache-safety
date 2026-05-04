from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json

ACTIVE_PRIMARY_DIR = Path("paper/generated/active_primary")
ACTIVE_CAUSAL_DIR = Path("paper/generated/active_causal")

PRIMARY_GENERATED_FILES = [
    "result_macros.tex",
    "main_results_table.tex",
    "suite_level_effects_table.tex",
]
CAUSAL_GENERATED_FILES = [
    "result_macros.tex",
    "causal_restoration_table.tex",
]
PRIMARY_FIGURES = [
    "safety_capability_phase_portrait.pdf",
    "selective_safety_erasure_heatmap.pdf",
    "prompt_effect_constellation.pdf",
    "cache_state_fingerprint.pdf",
    "safety_state_atlas.pdf",
]
CAUSAL_FIGURES = [
    "causal_restoration_fraction.pdf",
    "causal_restoration_flow.pdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync selected generated tables/macros and figures into stable active "
            "paths consumed by the LaTeX manuscript."
        )
    )
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--causal-results-dir", type=Path, required=True)
    parser.add_argument("--primary-generated-dir", type=Path, required=True)
    parser.add_argument("--causal-generated-dir", type=Path, required=True)
    parser.add_argument("--active-primary-dir", type=Path, default=ACTIVE_PRIMARY_DIR)
    parser.add_argument("--active-causal-dir", type=Path, default=ACTIVE_CAUSAL_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected paper asset is missing.",
    )
    args = parser.parse_args()

    missing = sync_active_paper_assets(
        primary_results_dir=args.primary_results_dir,
        causal_results_dir=args.causal_results_dir,
        primary_generated_dir=args.primary_generated_dir,
        causal_generated_dir=args.causal_generated_dir,
        active_primary_dir=args.active_primary_dir,
        active_causal_dir=args.active_causal_dir,
    )
    if missing and args.strict:
        for path in missing:
            print(f"Missing active paper asset source: {path}")
        raise SystemExit(1)
    print(
        "Synced active paper assets: "
        f"{args.active_primary_dir} and {args.active_causal_dir}"
    )


def sync_active_paper_assets(
    *,
    primary_results_dir: Path,
    causal_results_dir: Path,
    primary_generated_dir: Path,
    causal_generated_dir: Path,
    active_primary_dir: Path = ACTIVE_PRIMARY_DIR,
    active_causal_dir: Path = ACTIVE_CAUSAL_DIR,
) -> list[str]:
    primary_missing = _sync_one(
        kind="primary",
        generated_dir=primary_generated_dir,
        results_dir=primary_results_dir,
        active_dir=active_primary_dir,
        generated_files=PRIMARY_GENERATED_FILES,
        figure_files=PRIMARY_FIGURES,
    )
    causal_missing = _sync_one(
        kind="causal",
        generated_dir=causal_generated_dir,
        results_dir=causal_results_dir,
        active_dir=active_causal_dir,
        generated_files=CAUSAL_GENERATED_FILES,
        figure_files=CAUSAL_FIGURES,
    )
    return [*primary_missing, *causal_missing]


def _sync_one(
    *,
    kind: str,
    generated_dir: Path,
    results_dir: Path,
    active_dir: Path,
    generated_files: list[str],
    figure_files: list[str],
) -> list[str]:
    if active_dir.exists():
        shutil.rmtree(active_dir)
    (active_dir / "figures").mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in generated_files:
        source = generated_dir / name
        target = active_dir / name
        _copy_if_present(source, target, copied, missing)
    for name in figure_files:
        source = results_dir / "figures" / name
        target = active_dir / "figures" / name
        _copy_if_present(source, target, copied, missing)

    manifest = {
        "schema_version": 1,
        "kind": kind,
        "active_dir": str(active_dir),
        "generated_dir": str(generated_dir),
        "results_dir": str(results_dir),
        "copied": copied,
        "missing": missing,
    }
    write_json(active_dir / "active_asset_manifest.json", manifest)
    return missing


def _copy_if_present(
    source: Path,
    target: Path,
    copied: list[dict[str, Any]],
    missing: list[str],
) -> None:
    if not source.exists():
        missing.append(str(source))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied.append(
        {
            "source": str(source),
            "target": str(target),
            "sha256": file_sha256(target),
            "bytes": target.stat().st_size,
        }
    )


if __name__ == "__main__":
    main()
