from __future__ import annotations

import argparse
import json
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

    metrics_path = args.results_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        selective_rows = [
            {
                "suite_policy": key,
                "suite": key.split("::", 1)[0],
                "policy": key.split("::", 1)[1],
                "index": value.get("selective_safety_erasure_index"),
                "safety_degradation": value.get("safety_degradation"),
                "capability_degradation": value.get("capability_degradation"),
            }
            for key, value in metrics.get("selective_safety_erasure", {}).items()
            if value.get("selective_safety_erasure_index") is not None
        ]
        if selective_rows:
            selective_df = pd.DataFrame(selective_rows)
            pivot = selective_df.pivot_table(
                index="suite", columns="policy", values="index", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), 4.5))
            im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
            ax.set_title("Selective Safety Erasure Index")
            fig.colorbar(im, ax=ax, label="safety degradation - capability degradation")
            fig.tight_layout()
            out = figures_dir / "selective_safety_erasure_heatmap.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            made.append(str(out))

            fig, ax = plt.subplots(figsize=(10, 5))
            top = selective_df.sort_values("index", ascending=False).head(12)
            labels = [f"{row.suite}\n{row.policy}" for row in top.itertuples()]
            ax.bar(labels, top["index"])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Largest Selective Safety Effects")
            ax.set_ylabel("Selective Safety Erasure Index")
            ax.tick_params(axis="x", labelrotation=45)
            fig.tight_layout()
            out = figures_dir / "top_selective_effects.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            made.append(str(out))

    cache_path = args.results_dir / "cache_stats.parquet"
    if cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
        if {"policy", "decode_step", "cache_l2_before", "cache_l2_after"}.issubset(cache_df.columns):
            cache_df = cache_df.copy()
            cache_df["l2_retained_fraction"] = cache_df["cache_l2_after"] / cache_df[
                "cache_l2_before"
            ].replace(0, float("nan"))
            grouped = (
                cache_df.groupby(["policy", "decode_step"])["l2_retained_fraction"]
                .mean()
                .reset_index()
            )
            if not grouped.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                for policy, policy_df in grouped.groupby("policy"):
                    ax.plot(
                        policy_df["decode_step"],
                        policy_df["l2_retained_fraction"],
                        label=policy,
                        alpha=0.85,
                    )
                ax.set_title("Cache L2 Retained Fraction Over Decoding")
                ax.set_xlabel("Decode step")
                ax.set_ylabel("L2 retained fraction")
                ax.set_ylim(0, 1.05)
                ax.legend()
                fig.tight_layout()
                out = figures_dir / "cache_l2_retained_fraction.png"
                fig.savefig(out, dpi=180)
                plt.close(fig)
                made.append(str(out))

    write_json(figures_dir / "manifest.json", {"figures": made})
    print(f"Wrote {len(made)} figure(s) to {figures_dir}")


if __name__ == "__main__":
    main()
