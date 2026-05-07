from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.metrics.aggregate import compute_run_metrics
from cache_safety_erasure.utils.io import file_sha256, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-model selectivity runs into one cross-family analysis artifact."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-dir", action="append", default=[], type=Path)
    parser.add_argument("--stage", default=None)
    args = parser.parse_args()

    run_dirs = list(args.run_dir)
    if not run_dirs and args.stage:
        run_dirs = sorted(Path("results").glob(f"selectivity_h200_{args.stage}_*"))
    run_dirs = [path for path in run_dirs if (path / "generations.jsonl").exists()]
    if not run_dirs:
        raise SystemExit("No completed selectivity run directories found to merge.")

    merged = merge_runs(run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "generations.jsonl", merged["generations"])
    write_json(args.output_dir / "metrics.json", compute_run_metrics(merged["generations"]))
    write_json(args.output_dir / "manifest.json", merged["manifest"])
    if merged["base_model_scores"]:
        write_jsonl(args.output_dir / "base_model_scores.jsonl", merged["base_model_scores"])
        write_json(
            args.output_dir / "base_model_metrics.json",
            compute_base_panel_metrics(merged["base_model_scores"]),
        )
    print(f"Merged {len(run_dirs)} selectivity run(s) into {args.output_dir}")


def merge_runs(run_dirs: list[Path]) -> dict[str, Any]:
    generations: list[dict[str, Any]] = []
    base_scores: list[dict[str, Any]] = []
    source_runs = []
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        source = {
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "model_id": manifest.get("model_id"),
            "model_family": manifest.get("model_family"),
            "model_track": manifest.get("model_track"),
            "git_commit": manifest.get("git_commit"),
            "git_dirty": manifest.get("git_dirty"),
            "manifest_sha256": file_sha256(run_dir / "manifest.json"),
            "generations_sha256": file_sha256(run_dir / "generations.jsonl"),
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
        }
        source_runs.append(source)
        for row in read_jsonl(run_dir / "generations.jsonl"):
            generations.append(
                {
                    **row,
                    "source_run_id": run_dir.name,
                    "source_run_dir": str(run_dir),
                }
            )
        base_path = run_dir / "base_model_scores.jsonl"
        if base_path.exists():
            for row in read_jsonl(base_path):
                base_scores.append(
                    {
                        **row,
                        "source_run_id": run_dir.name,
                        "source_run_dir": str(run_dir),
                    }
                )
    manifest = {
        "schema_version": 1,
        "source_run_count": len(source_runs),
        "source_runs": source_runs,
        "generation_count": len(generations),
        "base_model_score_count": len(base_scores),
        "instruction_tuned_families": sorted(
            {
                str(run["model_family"])
                for run in source_runs
                if run.get("model_track") == "chat_safety" and run.get("model_family")
            }
        ),
    }
    return {"generations": generations, "base_model_scores": base_scores, "manifest": manifest}


def compute_base_panel_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from score_base_model_track import compute_base_model_metrics

    return compute_base_model_metrics(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
