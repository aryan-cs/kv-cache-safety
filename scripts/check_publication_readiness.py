from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a result directory is paper-ready.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--min-prompts-per-suite", type=int, default=100)
    parser.add_argument("--max-ci-width", type=float, default=0.08)
    args = parser.parse_args()

    failures: list[str] = []
    generations = args.results_dir / "generations.jsonl"
    metrics_path = args.results_dir / "metrics.json"
    for required in [
        "config.resolved.yaml",
        "environment.json",
        "manifest.json",
        "prompts.jsonl",
        "generations.jsonl",
        "metrics.json",
        "cache_stats.parquet",
    ]:
        if not (args.results_dir / required).exists():
            failures.append(f"missing artifact: {required}")

    if generations.exists():
        counts: dict[str, set[str]] = {}
        with generations.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts.setdefault(row["suite"], set()).add(row["prompt_id"])
        for suite, prompt_ids in counts.items():
            if len(prompt_ids) < args.min_prompts_per_suite:
                failures.append(
                    f"suite `{suite}` has {len(prompt_ids)} prompts; need >= {args.min_prompts_per_suite}"
                )

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        for key, value in metrics.get("selective_safety_erasure", {}).items():
            ci = value.get("paired_safety_degradation_ci", {})
            if ci.get("ci_low") is None or ci.get("ci_high") is None:
                failures.append(f"{key}: missing paired safety CI")
                continue
            width = ci["ci_high"] - ci["ci_low"]
            if width > args.max_ci_width:
                failures.append(
                    f"{key}: paired safety CI width {width:.3f}; target <= {args.max_ci_width:.3f}"
                )

    if failures:
        print("NOT PAPER READY")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PAPER READY CHECK PASSED")


if __name__ == "__main__":
    main()
