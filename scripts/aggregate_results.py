from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.metrics.aggregate import compute_run_metrics
from cache_safety_erasure.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate generation metrics for a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    generations = args.results_dir / "generations.jsonl"
    if not generations.exists():
        raise SystemExit(f"Missing generations file: {generations}")
    metrics = compute_run_metrics(read_jsonl(generations))
    write_json(args.results_dir / "metrics.json", metrics)
    print(f"Wrote {args.results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
