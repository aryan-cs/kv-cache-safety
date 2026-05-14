from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import utc_timestamp, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a machine-readable H200-to-Mac selectivity handoff manifest."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.run_dir / "selectivity_handoff.json"
    handoff = build_handoff(
        run_dir=args.run_dir,
        config=args.config,
        stage=args.stage,
        model_key=args.model_key,
    )
    write_json(output, handoff)
    print(f"Wrote selectivity handoff: {output}")


def build_handoff(
    *,
    run_dir: Path,
    config: Path,
    stage: str,
    model_key: str,
) -> dict[str, Any]:
    run_id = run_dir.name
    progress = _read_json(run_dir / "progress.json")
    manifest = _read_json(run_dir / "manifest.json")
    expected = int(progress.get("expected") or manifest.get("expected_generation_count") or 0)
    completed = int(progress.get("completed") or _jsonl_row_count(run_dir / "generations.jsonl"))
    complete = expected > 0 and completed >= expected and progress.get("activity") == "complete"
    audit_prefix = f"docs/audit/{run_id}"
    return {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "run_id": run_id,
        "stage": stage,
        "model_key": model_key,
        "model_id": manifest.get("model_id"),
        "model_family": manifest.get("model_family"),
        "model_track": manifest.get("model_track"),
        "config": str(config),
        "remote_run_dir": f"results/{run_id}",
        "complete": complete,
        "completed": completed,
        "expected": expected,
        "progress": progress,
        "fetch_command": [
            "bash",
            "scripts/fetch_h200_selectivity_results.sh",
            f"results/{run_id}",
        ],
        "local_judging_command": [
            "uv",
            "run",
            "python",
            "scripts/sync_and_judge_selectivity_run.py",
            "--run-id",
            run_id,
            "--workers",
            "4",
        ],
        "local_judging_outputs": {
            "audit_csv": f"{audit_prefix}_audit_blinded.csv",
            "audit_key": f"{audit_prefix}_audit_key.jsonl",
            "approved_input": f"{audit_prefix}_audit_key.gemini_approved.jsonl",
            "judgments": f"{audit_prefix}_judgments.gemini.jsonl",
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
