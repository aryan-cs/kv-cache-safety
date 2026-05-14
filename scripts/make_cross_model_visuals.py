"""Cross-model visualization gallery from Gemini-judged selectivity audits.

Reads every ``docs/audit/*_judgments.gemini.jsonl`` and emits a varied set of
figures into ``docs/generated/cross_model_visuals``. The intent is breadth: many
candidates so the reviewer can keep the strong ones and drop the rest.

Run: ``uv run python scripts/make_cross_model_visuals.py``
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_FIELDS = (
    "refusal_correct",
    "unsafe_compliance",
    "system_leakage",
    "over_refusal",
    "capability_correct",
)
POLICY_ORDER = (
    "none",
    "sliding_window__budget128",
    "sink_recent__budget128__sink8",
    "policy_pinned__budget128__sink8",
    "user_pinned__budget128__sink8",
    "random_matched__budget128__seed991",
)
PALETTE = {
    "qwen2_5_7b_instruct": "#2563eb",
    "qwen2_5_7b_base": "#60a5fa",
    "qwen3_5_9b": "#1e3a8a",
    "llama3_1_8b_instruct": "#dc2626",
    "gemma2_9b_it": "#0ea5e9",
    "mistral_7b_instruct_v0_3": "#f59e0b",
    "olmo3_7b_instruct": "#10b981",
    "phi4": "#a855f7",
    "gpt_oss_20b": "#475569",
}


def load_judgments(audit_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_dir.glob("*_judgments.gemini.jsonl")):
        model_key = path.stem.removesuffix("_judgments.gemini").removeprefix(
            "selectivity_h200_powered_"
        )
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                labels = r.get("labels") or {}
                parsed = r.get("parser_status") == "parsed"
                base = {
                    "model_key": model_key,
                    "model_id": r.get("model_id", ""),
                    "evaluated_family": r.get("evaluated_family", ""),
                    "suite": r.get("suite", ""),
                    "policy": r.get("policy", ""),
                    "category": r.get("category", ""),
                    "audit_id": r.get("audit_id", ""),
                    "parser_status": r.get("parser_status", ""),
                    "confidence": float(labels.get("confidence") or 0.0),
                    "response_length_words": r.get("response_length_words", 0),
                    "_dropped": not parsed,
                }
                for field in LABEL_FIELDS:
                    val = labels.get(field) if parsed else None
                    if val is None:
                        base[field] = float("nan")
                    else:
                        base[field] = float(bool(val))
                rows.append(base)
    df = pd.DataFrame(rows)
    for field in LABEL_FIELDS:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    return df


def short_policy(policy: str) -> str:
    if policy == "none":
        return "none"
    return re.sub(r"__budget\d+(?:__seed\d+)?(?:__sink\d+)?", "", policy)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def model_color(model_key: str) -> str:
    return PALETTE.get(model_key, "#374151")


def fig_safety_radar(df: pd.DataFrame, out: Path) -> None:
    metrics = ["refusal_correct", "capability_correct", "unsafe_compliance", "system_leakage", "over_refusal"]
    invert = {"unsafe_compliance", "system_leakage", "over_refusal"}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for model_key, group in df[~df["_dropped"]].groupby("model_key"):
        values = []
        for m in metrics:
            col = group[m].dropna()
            if col.empty:
                values.append(0.5)
                continue
            rate = col.astype(float).mean()
            values.append(1.0 - rate if m in invert else rate)
        values += values[:1]
        ax.plot(angles, values, label=model_key, color=model_color(model_key), linewidth=2, marker="o")
        ax.fill(angles, values, color=model_color(model_key), alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ") + ("\n(inverted)" if m in invert else "") for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("Per-model safety profile (higher = better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_label_heatmap(df: pd.DataFrame, out: Path) -> None:
    valid = df[~df["_dropped"]]
    pivot = (
        valid.groupby(["model_key", "suite"])[list(LABEL_FIELDS)]
        .mean()
        .reset_index()
    )
    if pivot.empty:
        return
    metric = "refusal_correct"
    table = pivot.pivot(index="model_key", columns="suite", values=metric)
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * table.shape[1] + 4), max(4, 0.4 * table.shape[0] + 2)))
    im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=35, ha="right")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black" if v > 0.5 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="refusal_correct rate")
    ax.set_title("Refusal correctness by model × suite")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_policy_dumbbell(df: pd.DataFrame, out: Path, metric: str = "refusal_correct") -> None:
    valid = df[~df["_dropped"]]
    by_mp = valid.groupby(["model_key", "policy"])[metric].mean().reset_index()
    none_rates = by_mp[by_mp["policy"] == "none"].set_index("model_key")[metric]
    policies = [p for p in POLICY_ORDER if p in set(by_mp["policy"]) and p != "none"]
    if not policies:
        return
    fig, axes = plt.subplots(1, len(policies), figsize=(2.5 * len(policies) + 2, 0.4 * by_mp["model_key"].nunique() + 3), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    for ax, policy in zip(axes, policies, strict=True):
        treat = by_mp[by_mp["policy"] == policy].set_index("model_key")[metric]
        rows = sorted(treat.index.intersection(none_rates.index))
        for i, m in enumerate(rows):
            ax.plot([none_rates[m], treat[m]], [i, i], color="#9ca3af", linewidth=2, zorder=1)
            ax.scatter(none_rates[m], i, color="#4b5563", s=40, zorder=2, label="baseline (none)" if i == 0 else None)
            ax.scatter(treat[m], i, color=model_color(m), s=70, zorder=3, label=m if i == 0 else None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(metric)
        ax.set_title(short_policy(policy), fontsize=10)
        ax.grid(axis="x", color="#e5e7eb")
    fig.suptitle(f"{metric} shift: cache-policy treatment vs baseline (none)", y=1.02)
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_unsafe_vs_overrefusal(df: pd.DataFrame, out: Path) -> None:
    valid = df[~df["_dropped"]]
    grouped = valid.groupby("model_key")[["unsafe_compliance", "over_refusal", "audit_id"]].agg(
        unsafe=("unsafe_compliance", "mean"),
        over=("over_refusal", "mean"),
        n=("audit_id", "size"),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    for m, row in grouped.iterrows():
        ax.scatter(row["unsafe"], row["over"], s=80 + math.log1p(row["n"]) * 40, color=model_color(m), edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.annotate(m, (row["unsafe"], row["over"]), fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.axhline(0.1, color="#d4d4d4", linestyle="--", linewidth=0.7)
    ax.axvline(0.1, color="#d4d4d4", linestyle="--", linewidth=0.7)
    ax.set_xlabel("unsafe_compliance rate (lower = safer)")
    ax.set_ylabel("over_refusal rate (lower = more helpful)")
    ax.set_title("Helpfulness vs safety trade-off (size ∝ sample count)")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_parser_status_bars(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby(["model_key", "parser_status"]).size().unstack(fill_value=0)
    counts = counts.div(counts.sum(axis=1), axis=0)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * counts.shape[0] + 2)))
    bottom = np.zeros(counts.shape[0])
    colors = {"parsed": "#16a34a", "parse_error": "#f59e0b", "blocked": "#dc2626", "unlabeled": "#6b7280"}
    for status in counts.columns:
        c = colors.get(status, "#9ca3af")
        ax.barh(counts.index, counts[status].values, left=bottom, color=c, edgecolor="white", label=status)
        bottom += counts[status].values
    ax.set_xlabel("fraction of judge attempts")
    ax.set_title("Judge parser status by model")
    ax.legend(loc="lower right")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_confidence_violin(df: pd.DataFrame, out: Path) -> None:
    valid = df[~df["_dropped"] & df["confidence"].notna()]
    models = sorted(valid["model_key"].unique())
    data = [valid[valid["model_key"] == m]["confidence"].astype(float).values for m in models]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(models) + 3), 5))
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(model_color(models[i]))
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts.get(key, None) and parts[key].set_color("#1f2937")
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("judge confidence")
    ax.set_title("Judge confidence distribution by model")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_suite_policy_smallmultiples(df: pd.DataFrame, out: Path, metric: str = "refusal_correct") -> None:
    valid = df[~df["_dropped"]]
    suites = sorted(valid["suite"].unique())
    if not suites:
        return
    cols = 3
    rows = math.ceil(len(suites) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4 + 1, rows * 3 + 1), sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    for i, suite in enumerate(suites):
        ax = axes[i // cols, i % cols]
        sub = valid[valid["suite"] == suite]
        pivot = sub.groupby(["model_key", "policy"])[metric].mean().reset_index()
        models = sorted(pivot["model_key"].unique())
        for model_key in models:
            mdata = pivot[pivot["model_key"] == model_key]
            xs = [short_policy(p) for p in mdata["policy"]]
            ax.plot(xs, mdata[metric], marker="o", color=model_color(model_key), label=model_key, linewidth=1.5)
        ax.set_title(suite, fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(color="#f1f5f9")
    for j in range(len(suites), rows * cols):
        axes[j // cols, j % cols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle(f"{metric} across policies by suite (small multiples)", y=1.01)
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_response_length_density(df: pd.DataFrame, out: Path) -> None:
    valid = df[(~df["_dropped"]) & df["response_length_words"].notna()]
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, valid["response_length_words"].quantile(0.99), 40)
    for model_key, group in valid.groupby("model_key"):
        ax.hist(
            group["response_length_words"],
            bins=bins,
            histtype="step",
            color=model_color(model_key),
            linewidth=2,
            label=model_key,
            density=True,
        )
    ax.set_xlabel("response length (words)")
    ax.set_ylabel("density")
    ax.set_title("Generated response length distribution")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_policy_delta_heatmap(df: pd.DataFrame, out: Path, metric: str = "refusal_correct") -> None:
    valid = df[~df["_dropped"]]
    by_mp = valid.groupby(["model_key", "policy"])[metric].mean().unstack()
    if "none" not in by_mp.columns:
        return
    delta = by_mp.subtract(by_mp["none"], axis=0).drop(columns=["none"])
    delta = delta[[p for p in POLICY_ORDER if p in delta.columns]]
    if delta.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * delta.shape[1] + 3), max(3, 0.4 * delta.shape[0] + 2)))
    vmax = float(np.nanmax(np.abs(delta.values))) or 0.1
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(delta.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(delta.shape[1]))
    ax.set_xticklabels([short_policy(c) for c in delta.columns], rotation=30, ha="right")
    ax.set_yticks(range(delta.shape[0]))
    ax.set_yticklabels(delta.index)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"Δ {metric} vs baseline")
    ax.set_title(f"Cache-policy effect on {metric} (Δ vs none)")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_category_strip(df: pd.DataFrame, out: Path) -> None:
    valid = df[~df["_dropped"]]
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * valid["category"].nunique() + 3)))
    categories = sorted(valid["category"].unique())
    rng = np.random.default_rng(0)
    for i, cat in enumerate(categories):
        sub = valid[valid["category"] == cat]
        for model_key, mgroup in sub.groupby("model_key"):
            rate = mgroup["refusal_correct"].dropna().astype(float).mean()
            jitter = rng.uniform(-0.2, 0.2)
            ax.scatter(rate, i + jitter, color=model_color(model_key), s=60, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("refusal_correct rate")
    ax.set_title("Per-category refusal accuracy strip plot (model points jittered)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=model_color(m), markersize=8, label=m) for m in sorted(valid["model_key"].unique())]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_treatment_arrows(df: pd.DataFrame, out: Path, metric: str = "unsafe_compliance") -> None:
    valid = df[~df["_dropped"]]
    by_mp = valid.groupby(["model_key", "policy"])[metric].mean().unstack()
    if "none" not in by_mp.columns:
        return
    policies = [p for p in POLICY_ORDER if p in by_mp.columns and p != "none"]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * by_mp.shape[0] + 3)))
    models = sorted(by_mp.index)
    for i, m in enumerate(models):
        baseline = by_mp.loc[m, "none"]
        if pd.isna(baseline):
            continue
        ax.scatter(baseline, i, color="#374151", s=70, zorder=3)
        for policy in policies:
            v = by_mp.loc[m, policy]
            if pd.isna(v):
                continue
            ax.annotate(
                "",
                xy=(v, i),
                xytext=(baseline, i),
                arrowprops=dict(arrowstyle="->", color=model_color(m), alpha=0.55, lw=1.2),
            )
            ax.scatter(v, i, color=model_color(m), s=18, alpha=0.7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel(metric)
    ax.set_xlim(0, max(0.05, by_mp.max().max() * 1.2))
    ax.set_title(f"Policy-induced shifts in {metric} (arrows from baseline)")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_safety_capability_quadrant(df: pd.DataFrame, out: Path) -> None:
    valid = df[~df["_dropped"]]
    grouped = valid.groupby("model_key").agg(
        safety=("unsafe_compliance", lambda s: 1.0 - s.astype(float).mean()),
        cap=("capability_correct", lambda s: s.dropna().astype(float).mean() if s.notna().any() else np.nan),
        n=("audit_id", "size"),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    for m, row in grouped.iterrows():
        if pd.isna(row["cap"]):
            continue
        ax.scatter(row["cap"], row["safety"], s=120 + math.log1p(row["n"]) * 40, color=model_color(m), edgecolor="black", linewidth=0.5)
        ax.annotate(m, (row["cap"], row["safety"]), fontsize=9, xytext=(6, 4), textcoords="offset points")
    ax.axhline(0.9, color="#d4d4d4", linestyle="--", linewidth=0.7)
    ax.axvline(0.5, color="#d4d4d4", linestyle="--", linewidth=0.7)
    ax.set_xlabel("capability_correct rate")
    ax.set_ylabel("1 - unsafe_compliance (safety)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Capability vs safety quadrant (target: top-right)")
    fig.savefig(out, format="png")
    plt.close(fig)


def fig_disagreement_summary(df: pd.DataFrame, out: Path) -> None:
    valid = df[~df["_dropped"]]
    summary = (
        valid.groupby("model_key")
        .apply(lambda g: pd.Series({
            "refusal_correct_mean": g["refusal_correct"].astype("float").mean(),
            "ambiguous_share": g["confidence"].astype("float").lt(0.6).mean(),
            "n": len(g),
        }), include_groups=False)
    )
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(summary) + 2)))
    y = np.arange(len(summary))
    ax.barh(y - 0.18, summary["refusal_correct_mean"], height=0.32, color="#16a34a", label="refusal_correct mean")
    ax.barh(y + 0.18, summary["ambiguous_share"], height=0.32, color="#a855f7", label="judge low-confidence share (<0.6)")
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("rate")
    ax.set_title("Per-model overview: refusal correctness vs judge uncertainty")
    ax.legend(loc="lower right")
    fig.savefig(out, format="png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("docs/audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/generated/cross_model_visuals"))
    args = parser.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_judgments(args.audit_dir)
    if df.empty:
        raise SystemExit("No judgments found in docs/audit; nothing to render.")

    valid = df[~df["_dropped"]]
    summary = {
        "models": sorted(df["model_key"].unique()),
        "total_judgment_rows": int(len(df)),
        "parsed_rows": int((~df["_dropped"]).sum()),
        "blocked_or_unparsed_rows": int(df["_dropped"].sum()),
        "suites": sorted(valid["suite"].unique().tolist()) if not valid.empty else [],
        "policies": sorted(valid["policy"].unique().tolist()) if not valid.empty else [],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fig_safety_radar(df, args.output_dir / "01_safety_radar.png")
    fig_label_heatmap(df, args.output_dir / "02_refusal_heatmap_model_x_suite.png")
    fig_policy_dumbbell(df, args.output_dir / "03_policy_dumbbell_refusal.png", metric="refusal_correct")
    fig_unsafe_vs_overrefusal(df, args.output_dir / "04_unsafe_vs_overrefusal_scatter.png")
    fig_parser_status_bars(df, args.output_dir / "05_judge_parser_status.png")
    fig_confidence_violin(df, args.output_dir / "06_judge_confidence_violin.png")
    fig_suite_policy_smallmultiples(df, args.output_dir / "07_suite_policy_small_multiples.png")
    fig_response_length_density(df, args.output_dir / "08_response_length_density.png")
    fig_policy_delta_heatmap(df, args.output_dir / "09_policy_delta_heatmap_refusal.png", metric="refusal_correct")
    fig_policy_delta_heatmap(df, args.output_dir / "10_policy_delta_heatmap_unsafe.png", metric="unsafe_compliance")
    fig_category_strip(df, args.output_dir / "11_category_strip.png")
    fig_treatment_arrows(df, args.output_dir / "12_treatment_arrows_unsafe.png", metric="unsafe_compliance")
    fig_treatment_arrows(df, args.output_dir / "13_treatment_arrows_leakage.png", metric="system_leakage")
    fig_safety_capability_quadrant(df, args.output_dir / "14_safety_capability_quadrant.png")
    fig_disagreement_summary(df, args.output_dir / "15_overview_disagreement.png")

    print(f"Wrote {summary['parsed_rows']} parsed judgments across {len(summary['models'])} models.")
    print(f"Figures: {args.output_dir}")


if __name__ == "__main__":
    main()
