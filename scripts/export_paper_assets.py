from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready result tables from a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()

    metrics_path = args.results_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    args.paper_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for policy, values in metrics.get("publication_summary", {}).get("policies", {}).items():
        contrast = metrics.get("policy_level_contrasts", {}).get(policy, {})
        ssei_ci = contrast.get("selective_safety_erasure_index_ci", {})
        summary_rows.append(
            {
                "policy": policy,
                "mean_safety_score": values.get("mean_safety_score"),
                "mean_capability_score": values.get("mean_capability_score"),
                "global_safety_degradation": values.get("global_safety_degradation"),
                "global_capability_degradation": values.get("global_capability_degradation"),
                "global_selective_safety_erasure_index": values.get(
                    "global_selective_safety_erasure_index"
                ),
                "policy_level_ssei": contrast.get("selective_safety_erasure_index"),
                "policy_level_ssei_ci_low": ssei_ci.get("ci_low"),
                "policy_level_ssei_ci_high": ssei_ci.get("ci_high"),
                "policy_level_safety_clusters": ssei_ci.get("n_safety"),
                "policy_level_capability_clusters": ssei_ci.get("n_capability"),
            }
        )
    write_markdown_table(
        args.paper_dir / "main_results_table.md",
        [
            "policy",
            "mean_safety_score",
            "mean_capability_score",
            "policy_level_ssei",
            "policy_level_ssei_ci_low",
            "policy_level_ssei_ci_high",
            "policy_level_safety_clusters",
            "policy_level_capability_clusters",
        ],
        summary_rows,
    )

    selective_rows = []
    for key, values in metrics.get("selective_safety_erasure", {}).items():
        suite, policy = key.split("::", 1)
        selective_rows.append(
            {
                "suite": suite,
                "policy": policy,
                "safety_degradation": values.get("safety_degradation"),
                "capability_degradation": values.get("capability_degradation"),
                "within_suite_ssei_if_capability_available": values.get(
                    "selective_safety_erasure_index"
                ),
                "paired_n": values.get("paired_safety_degradation_ci", {}).get("paired_n"),
                "cluster_n": values.get("paired_safety_degradation_ci", {}).get("cluster_n"),
                "safety_ci_low": values.get("paired_safety_degradation_ci", {}).get("ci_low"),
                "safety_ci_high": values.get("paired_safety_degradation_ci", {}).get("ci_high"),
            }
        )
    write_markdown_table(
        args.paper_dir / "suite_level_effects_table.md",
        [
            "suite",
            "policy",
            "safety_degradation",
            "capability_degradation",
            "within_suite_ssei_if_capability_available",
            "paired_n",
            "cluster_n",
            "safety_ci_low",
            "safety_ci_high",
        ],
        selective_rows,
    )
    restoration_rows = []
    for key, values in metrics.get("causal_restoration", {}).items():
        suite, policy = key.split("::", 1)
        restoration_rows.append(
            {
                "suite": suite,
                "policy": policy,
                "compressed_policy": values.get("compressed_policy"),
                "safety_restoration_fraction": values.get("safety_restoration_fraction"),
                "refusal_restoration_fraction": values.get("refusal_restoration_fraction"),
                "leakage_avoidance_restoration_fraction": values.get(
                    "leakage_avoidance_restoration_fraction"
                ),
            }
        )
    write_markdown_table(
        args.paper_dir / "causal_restoration_table.md",
        [
            "suite",
            "policy",
            "compressed_policy",
            "safety_restoration_fraction",
            "refusal_restoration_fraction",
            "leakage_avoidance_restoration_fraction",
        ],
        restoration_rows,
    )
    print(f"Wrote paper tables to {args.paper_dir}")


def write_markdown_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
