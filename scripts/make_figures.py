from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies with `uv sync --extra dev` to make figures.") from exc

    generations_path = args.results_dir / "generations.jsonl"
    if not generations_path.exists():
        raise SystemExit(f"Missing generations file: {generations_path}")
    df = pd.read_json(generations_path, lines=True)
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    made: list[str] = []
    for metric in ["safety_score", "capability_score", "rouge_l_leakage_recall"]:
        if metric not in df or df[metric].dropna().empty:
            continue
        grouped = df.groupby(["suite", "policy"], dropna=False)[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        for suite, suite_df in grouped.groupby("suite"):
            ax.plot(suite_df["policy"], suite_df[metric], marker="o", label=suite)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.set_xlabel("Cache policy")
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend()
        fig.tight_layout()
        out = figures_dir / f"{metric}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        made.append(str(out))

    write_json(figures_dir / "manifest.json", {"figures": made})
    print(f"Wrote {len(made)} figure(s) to {figures_dir}")


if __name__ == "__main__":
    main()
