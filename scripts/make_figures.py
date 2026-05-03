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

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_df = selective_df.dropna(subset=["safety_degradation"]).copy()
            plot_df["capability_degradation"] = plot_df["capability_degradation"].fillna(0.0)
            plot_df = plot_df.reset_index(drop=True)
            plot_df["x_plot"] = plot_df["capability_degradation"] + (
                (plot_df.index % 5) - 2
            ) * 0.002
            plot_df["y_plot"] = plot_df["safety_degradation"] + (
                ((plot_df.index // 5) % 5) - 2
            ) * 0.002
            ax.axline((0, 0), slope=1, color="0.6", linewidth=1, linestyle="--")
            for suite, suite_df in plot_df.groupby("suite"):
                ax.scatter(suite_df["x_plot"], suite_df["y_plot"], s=64, label=suite, alpha=0.85)
            labeled = plot_df[plot_df["index"].abs() > 0].copy()
            if not labeled.empty:
                labeled = labeled.sort_values("index", key=lambda series: series.abs(), ascending=False).head(8)
                for row in labeled.itertuples():
                    ax.annotate(
                        row.policy,
                        (row.x_plot, row.y_plot),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Safety vs Capability Degradation")
            ax.set_xlabel("Capability degradation")
            ax.set_ylabel("Safety degradation")
            ax.legend(title="suite", fontsize=8)
            fig.tight_layout()
            out = figures_dir / "safety_vs_capability_degradation.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            made.append(str(out))

            forest_rows = []
            for key, value in metrics.get("selective_safety_erasure", {}).items():
                ci = value.get("paired_safety_degradation_ci", {})
                if ci.get("mean") is None or ci.get("ci_low") is None or ci.get("ci_high") is None:
                    continue
                forest_rows.append(
                    {
                        "label": key.replace("::", "\n"),
                        "mean": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "cluster_n": ci.get("cluster_n"),
                    }
                )
            if forest_rows:
                forest_df = pd.DataFrame(forest_rows).sort_values("mean")
                fig_height = max(4, 0.35 * len(forest_df))
                fig, ax = plt.subplots(figsize=(9, fig_height))
                y = range(len(forest_df))
                xerr = [
                    forest_df["mean"] - forest_df["ci_low"],
                    forest_df["ci_high"] - forest_df["mean"],
                ]
                ax.errorbar(forest_df["mean"], y, xerr=xerr, fmt="o", capsize=3)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(list(y), labels=forest_df["label"])
                ax.set_xlabel("Paired safety degradation")
                ax.set_title("Paired Safety Degradation With Prompt-Clustered CIs")
                fig.tight_layout()
                out = figures_dir / "paired_safety_degradation_forest.png"
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
        role_columns = [col for col in cache_df.columns if col.startswith("retained_") and col.endswith("_tokens")]
        role_rows = []
        for retained_col in role_columns:
            role = retained_col[len("retained_") : -len("_tokens")]
            evicted_col = f"evicted_{role}_tokens"
            if evicted_col not in cache_df.columns:
                continue
            grouped = cache_df.groupby("policy")[[retained_col, evicted_col]].sum().reset_index()
            for row in grouped.itertuples(index=False):
                retained_count = float(getattr(row, retained_col))
                evicted_count = float(getattr(row, evicted_col))
                total = retained_count + evicted_count
                if total <= 0:
                    continue
                role_rows.append(
                    {
                        "policy": row.policy,
                        "role": role,
                        "retention_fraction": retained_count / total,
                    }
                )
        if role_rows:
            role_df = pd.DataFrame(role_rows)
            pivot = role_df.pivot_table(
                index="role", columns="policy", values="retention_fraction", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), 4.5))
            im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
            ax.set_title("Token Role Retention By Cache Policy")
            fig.colorbar(im, ax=ax, label="retained / observed role tokens")
            fig.tight_layout()
            out = figures_dir / "token_role_retention_heatmap.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            made.append(str(out))

    write_json(figures_dir / "manifest.json", {"figures": made})
    print(f"Wrote {len(made)} figure(s) to {figures_dir}")


if __name__ == "__main__":
    main()
