from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json

ROLE_ORDER = {
    "system": 0,
    "hidden_system": 1,
    "template": 2,
    "user": 3,
    "generated": 4,
    "special": 5,
    "unknown": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies with `uv sync --extra dev` to make figures.") from exc
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "cache-safety-erasure"
    plt.rcParams["pdf.fonttype"] = 42

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

            phase_df = _phase_portrait_rows(selective_df)
            if not phase_df.empty:
                fig, ax = plt.subplots(figsize=(9, 7))
                ax.axline((0, 0), slope=1, color="0.75", linewidth=1, linestyle="--", zorder=0)
                ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)
                ax.axvline(0, color="0.25", linewidth=0.8, zorder=0)
                cmap = plt.get_cmap("tab10")
                for color_idx, ((suite, family), group) in enumerate(
                    phase_df.groupby(["suite", "policy_family"], dropna=False)
                ):
                    group = group.sort_values(["budget_sort", "policy"])
                    color = cmap(color_idx % 10)
                    ax.plot(
                        group["capability_degradation"],
                        group["safety_degradation"],
                        marker="o",
                        linewidth=2.0,
                        markersize=6,
                        alpha=0.88,
                        color=color,
                        label=f"{suite} / {family}",
                    )
                    for start, end in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples(), strict=False):
                        ax.annotate(
                            "",
                            xy=(end.capability_degradation, end.safety_degradation),
                            xytext=(start.capability_degradation, start.safety_degradation),
                            arrowprops={
                                "arrowstyle": "->",
                                "color": color,
                                "lw": 1.5,
                                "alpha": 0.72,
                                "shrinkA": 6,
                                "shrinkB": 6,
                            },
                        )
                    for row in group.itertuples():
                        if row.budget_label:
                            ax.annotate(
                                row.budget_label,
                                (row.capability_degradation, row.safety_degradation),
                                xytext=(4, 3),
                                textcoords="offset points",
                                fontsize=7,
                                color=color,
                            )
                ax.fill_between(
                    [-1, 1],
                    [-1, 1],
                    [1, 3],
                    color="#d73027",
                    alpha=0.06,
                    label="selective safety-loss region",
                )
                ax.set_xlim(
                    min(-0.05, phase_df["capability_degradation"].min() - 0.05),
                    max(0.25, phase_df["capability_degradation"].max() + 0.05),
                )
                ax.set_ylim(
                    min(-0.05, phase_df["safety_degradation"].min() - 0.05),
                    max(0.25, phase_df["safety_degradation"].max() + 0.05),
                )
                ax.set_title("Safety-Capability Phase Portrait")
                ax.set_xlabel("Capability degradation")
                ax.set_ylabel("Safety degradation")
                ax.legend(fontsize=7, ncols=2)
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "safety_capability_phase_portrait",
                    made,
                    data_rows=phase_df.to_dict(orient="records"),
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

        restoration_rows = [
            {
                "suite_policy": key,
                "suite": key.split("::", 1)[0],
                "policy": key.split("::", 1)[1],
                "compressed_policy": value.get("compressed_policy"),
                "safety_restoration_fraction": value.get("safety_restoration_fraction"),
                "refusal_restoration_fraction": value.get("refusal_restoration_fraction"),
                "leakage_avoidance_restoration_fraction": value.get(
                    "leakage_avoidance_restoration_fraction"
                ),
            }
            for key, value in metrics.get("causal_restoration", {}).items()
        ]
        if restoration_rows:
            restoration_df = pd.DataFrame(restoration_rows)
            plot_df = restoration_df.dropna(subset=["safety_restoration_fraction"]).copy()
            if not plot_df.empty:
                plot_df = plot_df.sort_values("safety_restoration_fraction")
                fig_height = max(4, 0.35 * len(plot_df))
                fig, ax = plt.subplots(figsize=(10, fig_height))
                labels = [f"{row.suite}\n{row.policy}" for row in plot_df.itertuples()]
                ax.barh(labels, plot_df["safety_restoration_fraction"])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.axvline(1, color="0.6", linewidth=1, linestyle="--")
                ax.set_xlabel("(patched - compressed) / (baseline - compressed)")
                ax.set_title("Causal Restoration Fraction")
                fig.tight_layout()
                _save_figure(
                    fig,
                    figures_dir,
                    "causal_restoration_fraction",
                    made,
                    data_rows=restoration_rows,
                )
                plt.close(fig)

                flow_df = _restoration_flow_rows(plot_df)
                if not flow_df.empty:
                    fig_height = max(4.5, 0.42 * len(flow_df))
                    fig, ax = plt.subplots(figsize=(10, fig_height))
                    y_positions = list(range(len(flow_df)))
                    ax.axvline(0, color="0.2", linewidth=1)
                    ax.axvline(1, color="0.65", linewidth=1, linestyle="--")
                    ax.text(0, len(flow_df) + 0.15, "compressed", ha="center", va="bottom", fontsize=9)
                    ax.text(1, len(flow_df) + 0.15, "baseline", ha="center", va="bottom", fontsize=9)
                    for y, row in zip(y_positions, flow_df.itertuples(), strict=False):
                        color = _restoration_color(row.policy)
                        ax.annotate(
                            "",
                            xy=(row.safety_restoration_fraction, y),
                            xytext=(0, y),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "lw": 2.4,
                                "color": color,
                                "alpha": 0.82,
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                        )
                        ax.scatter(
                            [row.safety_restoration_fraction],
                            [y],
                            s=90,
                            color=color,
                            edgecolor="white",
                            linewidth=0.8,
                            zorder=3,
                        )
                    ax.set_yticks(y_positions, labels=flow_df["label"])
                    ax.set_xlim(-0.08, max(1.08, flow_df["safety_restoration_fraction"].max() + 0.08))
                    ax.set_ylim(-0.8, len(flow_df) + 0.6)
                    ax.set_xlabel("Restoration fraction: (patched - compressed) / (baseline - compressed)")
                    ax.set_title("Causal Restoration Flow")
                    fig.tight_layout()
                    _save_figure(
                        fig,
                        figures_dir,
                        "causal_restoration_flow",
                        made,
                        data_rows=flow_df.to_dict(orient="records"),
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
        fingerprint_rows = _stream_cache_fingerprint(
            cache_path,
            args.results_dir / "prompts.jsonl",
            bin_count=48,
        )
        if fingerprint_rows:
            fingerprint_df = pd.DataFrame(fingerprint_rows)
            fingerprint_df["row_label"] = fingerprint_df["policy"] + " / " + fingerprint_df["role"]
            row_order = (
                fingerprint_df[["row_label", "policy", "role"]]
                .drop_duplicates()
                .sort_values(
                    by=["policy", "role"],
                    key=lambda series: series.map(lambda value: ROLE_ORDER.get(str(value), 99))
                    if series.name == "role"
                    else series,
                )
            )
            pivot = fingerprint_df.pivot_table(
                index="row_label",
                columns="token_bin",
                values="retention_fraction",
                aggfunc="mean",
            ).reindex(row_order["row_label"])
            fig_height = max(5, 0.24 * len(pivot.index))
            fig, ax = plt.subplots(figsize=(12, fig_height))
            im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="magma", vmin=0, vmax=1)
            ax.set_xticks(
                [0, max(0, pivot.shape[1] // 4), max(0, pivot.shape[1] // 2), max(0, 3 * pivot.shape[1] // 4), max(0, pivot.shape[1] - 1)],
                labels=["start", "25%", "50%", "75%", "end"],
            )
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=7)
            ax.set_xlabel("Normalized prompt-cache position")
            ax.set_title("Cache-State Fingerprint By Policy, Role, And Token Position")
            fig.colorbar(im, ax=ax, label="retained fraction")
            fig.tight_layout()
            _save_figure(
                fig,
                figures_dir,
                "cache_state_fingerprint",
                made,
                data_rows=fingerprint_rows,
            )
            plt.close(fig)

    write_json(
        figures_dir / "manifest.json",
        {
            "schema_version": 1,
            "source_artifacts": _source_artifacts(args.results_dir),
            "figures": made,
        },
    )
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
    pdf_path = figures_dir / f"{stem}.pdf"
    data_path = figures_dir / f"{stem}.csv"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    entry: dict[str, Any] = {
        "name": stem,
        "png": str(png_path),
        "png_sha256": file_sha256(png_path),
        "svg": str(svg_path),
        "svg_sha256": file_sha256(svg_path),
        "pdf": str(pdf_path),
        "pdf_sha256": file_sha256(pdf_path),
    }
    if data_rows is not None:
        _write_csv(data_path, data_rows)
        entry["data_csv"] = str(data_path)
        entry["data_csv_sha256"] = file_sha256(data_path)
        entry["data_row_count"] = len(data_rows)
    made.append(entry)


def _source_artifacts(results_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        path = results_dir / name
        artifacts[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return artifacts


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


def _phase_portrait_rows(selective_df: Any) -> Any:
    import pandas as pd

    rows = []
    for row in selective_df.to_dict(orient="records"):
        safety = _finite_float(row.get("safety_degradation"))
        capability = _finite_float(row.get("capability_degradation"))
        if safety is None or capability is None:
            continue
        policy = str(row.get("policy"))
        family, budget_sort, budget_label = _policy_shape(policy)
        rows.append(
            {
                "suite": row.get("suite"),
                "policy": policy,
                "policy_family": family,
                "budget_sort": budget_sort,
                "budget_label": budget_label,
                "safety_degradation": safety,
                "capability_degradation": capability,
                "selective_safety_erasure_index": _finite_float(
                    row.get("selective_safety_erasure_index") or row.get("index")
                ),
            }
        )
    return pd.DataFrame(rows)


def _policy_shape(policy: str) -> tuple[str, float, str]:
    family = policy.split("__", 1)[0]
    budget_match = re.search(r"__budget(\d+)", policy)
    if budget_match:
        budget = float(budget_match.group(1))
        return family, budget, f"b={int(budget)}"
    if "int4" in policy:
        return family, 4.0, "4-bit"
    if "int8" in policy:
        return family, 8.0, "8-bit"
    if policy == "none":
        return family, 0.0, "base"
    return family, 1_000_000.0, ""


def _restoration_flow_rows(restoration_df: Any) -> Any:
    import pandas as pd

    rows = []
    for row in restoration_df.itertuples():
        fraction = _finite_float(row.safety_restoration_fraction)
        if fraction is None:
            continue
        rows.append(
            {
                "suite": row.suite,
                "policy": row.policy,
                "compressed_policy": row.compressed_policy,
                "safety_restoration_fraction": fraction,
                "label": f"{row.suite} / {_short_policy_label(row.policy)}",
            }
        )
    return pd.DataFrame(rows).sort_values("safety_restoration_fraction") if rows else pd.DataFrame()


def _short_policy_label(policy: str) -> str:
    label = policy.replace("__", " / ")
    label = label.replace("patchkey-value", "patch K,V")
    label = label.replace("rolesystem", "system")
    label = label.replace("roleuser", "user")
    return label


def _restoration_color(policy: str) -> str:
    if "rolesystem" in policy or "system" in policy:
        return "#1b9e77"
    if "roleuser" in policy or "matchsystem" in policy:
        return "#7570b3"
    if "policy_pinned" in policy:
        return "#d95f02"
    return "#4d4d4d"


def _stream_cache_fingerprint(
    cache_path: Path,
    prompts_path: Path,
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required to summarize cache_stats.parquet.") from exc
    if not prompts_path.exists():
        return []
    prompt_roles = _load_prompt_roles(prompts_path)
    parquet_file = pq.ParquetFile(cache_path)
    required = {
        "prompt_id",
        "policy",
        "decode_step",
        "original_seq_len",
        "retained_indices",
        "evicted_indices",
    }
    if not required.issubset(set(parquet_file.schema.names)):
        return []
    counts: dict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    columns = sorted(required)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=50_000):
        table = batch.to_pydict()
        for idx, raw_policy in enumerate(table.get("policy", [])):
            if int(_float_at(table, "decode_step", idx)) != 0:
                continue
            prompt_id = str(table["prompt_id"][idx])
            policy = str(raw_policy)
            seq_len = int(_float_at(table, "original_seq_len", idx))
            if seq_len <= 0:
                continue
            roles = prompt_roles.get(prompt_id, [])
            for token_idx in _parse_indices(table["retained_indices"][idx]):
                role = _role_at(roles, token_idx)
                token_bin = min(bin_count - 1, int((token_idx / seq_len) * bin_count))
                counts[(policy, role, token_bin)][0] += 1.0
            for token_idx in _parse_indices(table["evicted_indices"][idx]):
                role = _role_at(roles, token_idx)
                token_bin = min(bin_count - 1, int((token_idx / seq_len) * bin_count))
                counts[(policy, role, token_bin)][1] += 1.0
    rows = []
    for (policy, role, token_bin), (retained, evicted) in sorted(
        counts.items(), key=lambda item: (item[0][0], ROLE_ORDER.get(item[0][1], 99), item[0][2])
    ):
        total = retained + evicted
        if total <= 0:
            continue
        rows.append(
            {
                "policy": policy,
                "role": role,
                "token_bin": token_bin,
                "retained_count": retained,
                "evicted_count": evicted,
                "retention_fraction": retained / total,
            }
        )
    return rows


def _load_prompt_roles(prompts_path: Path) -> dict[str, list[str]]:
    roles_by_prompt: dict[str, list[str]] = {}
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rendered = record.get("rendered_prompt") or {}
            roles = rendered.get("token_roles") or []
            if roles:
                roles_by_prompt[str(record.get("prompt_id"))] = [str(role) for role in roles]
    return roles_by_prompt


def _parse_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def _role_at(roles: list[str], idx: int) -> str:
    if 0 <= idx < len(roles):
        return roles[idx]
    return "unknown"


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


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
