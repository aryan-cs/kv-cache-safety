from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json


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

    made: list[dict[str, Any]] = []
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
        _save_figure(
            fig,
            figures_dir,
            metric,
            made,
            data_rows=grouped.to_dict(orient="records"),
        )
        plt.close(fig)

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
            _save_figure(
                fig,
                figures_dir,
                "selective_safety_erasure_heatmap",
                made,
                data_rows=selective_df.to_dict(orient="records"),
            )
            plt.close(fig)

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
            _save_figure(
                fig,
                figures_dir,
                "safety_vs_capability_degradation",
                made,
                data_rows=plot_df.to_dict(orient="records"),
            )
            plt.close(fig)

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
                _save_figure(
                    fig,
                    figures_dir,
                    "paired_safety_degradation_forest",
                    made,
                    data_rows=forest_rows,
                )
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            top = selective_df.sort_values("index", ascending=False).head(12)
            labels = [f"{row.suite}\n{row.policy}" for row in top.itertuples()]
            ax.bar(labels, top["index"])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Largest Selective Safety Effects")
            ax.set_ylabel("Selective Safety Erasure Index")
            ax.tick_params(axis="x", labelrotation=45)
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "top_selective_effects",
                made,
                data_rows=top.to_dict(orient="records"),
            )
            plt.close(fig)

    cache_path = args.results_dir / "cache_stats.parquet"
    if cache_path.exists():
        cache_summaries = _stream_cache_summaries(cache_path)
        l2_grouped = cache_summaries["l2_rows"]
        if l2_grouped:
            grouped = pd.DataFrame(l2_grouped)
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
                _save_figure(
                    fig,
                    figures_dir,
                    "cache_l2_retained_fraction",
                    made,
                    data_rows=l2_grouped,
                )
                plt.close(fig)
        role_rows = cache_summaries["role_rows"]
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
            _save_figure(
                fig,
                figures_dir,
                "token_role_retention_heatmap",
                made,
                data_rows=role_rows,
            )
            plt.close(fig)

    write_json(figures_dir / "manifest.json", {"figures": made})
    print(f"Wrote {len(made)} figure(s) to {figures_dir}")


def _save_figure(
    fig: Any,
    figures_dir: Path,
    stem: str,
    made: list[dict[str, Any]],
    *,
    data_rows: list[dict[str, Any]] | None = None,
) -> None:
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    data_path = figures_dir / f"{stem}.csv"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    entry: dict[str, Any] = {
        "name": stem,
        "png": str(png_path),
        "png_sha256": file_sha256(png_path),
        "svg": str(svg_path),
        "svg_sha256": file_sha256(svg_path),
    }
    if data_rows is not None:
        _write_csv(data_path, data_rows)
        entry["data_csv"] = str(data_path)
        entry["data_csv_sha256"] = file_sha256(data_path)
        entry["data_row_count"] = len(data_rows)
    made.append(entry)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stream_cache_summaries(cache_path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required to summarize cache_stats.parquet.") from exc
    parquet_file = pq.ParquetFile(cache_path)
    schema_names = set(parquet_file.schema.names)
    l2_columns = [
        column
        for column in ["policy", "decode_step", "cache_l2_before", "cache_l2_after"]
        if column in schema_names
    ]
    role_columns = [
        column
        for column in schema_names
        if (
            column.startswith("retained_")
            and column.endswith("_tokens")
            and f"evicted_{column[len('retained_'):]}" in schema_names
        )
    ]
    columns = sorted(set(l2_columns + role_columns + ["policy"] + [
        f"evicted_{column[len('retained_'):]}" for column in role_columns
    ]))
    if "policy" not in columns:
        return {"l2_rows": [], "role_rows": []}

    l2_sums: dict[tuple[str, int], list[float]] = {}
    role_sums: dict[tuple[str, str], list[float]] = {}
    for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
        table = batch.to_pydict()
        policies = table.get("policy", [])
        for idx, raw_policy in enumerate(policies):
            policy = str(raw_policy)
            if {"decode_step", "cache_l2_before", "cache_l2_after"}.issubset(table):
                before = _float_at(table, "cache_l2_before", idx)
                after = _float_at(table, "cache_l2_after", idx)
                if before:
                    key = (policy, int(_float_at(table, "decode_step", idx)))
                    l2_sums.setdefault(key, [0.0, 0.0])
                    l2_sums[key][0] += after / before
                    l2_sums[key][1] += 1.0
            for retained_col in role_columns:
                role = retained_col[len("retained_") : -len("_tokens")]
                evicted_col = f"evicted_{role}_tokens"
                retained = _float_at(table, retained_col, idx)
                evicted = _float_at(table, evicted_col, idx)
                if retained or evicted:
                    key = (policy, role)
                    role_sums.setdefault(key, [0.0, 0.0])
                    role_sums[key][0] += retained
                    role_sums[key][1] += evicted

    l2_rows = [
        {
            "policy": policy,
            "decode_step": decode_step,
            "l2_retained_fraction": total / count if count else None,
        }
        for (policy, decode_step), (total, count) in sorted(l2_sums.items())
    ]
    role_rows = []
    for (policy, role), (retained, evicted) in sorted(role_sums.items()):
        total = retained + evicted
        if total <= 0:
            continue
        role_rows.append(
            {
                "policy": policy,
                "role": role,
                "retention_fraction": retained / total,
                "retained_count": retained,
                "evicted_count": evicted,
            }
        )
    return {"l2_rows": l2_rows, "role_rows": role_rows}


def _float_at(table: dict[str, list[Any]], column: str, idx: int) -> float:
    values = table.get(column)
    if values is None:
        return 0.0
    value = values[idx]
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
