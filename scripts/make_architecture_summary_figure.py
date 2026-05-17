"""Architecture-summary scatter for the paper's failure-mode finding.

Plots each panel model on a 2D grid where:
  x-axis: attention scope. 0 = full-attention every layer, 1 = mixed local/global,
          2 = local sliding-window every layer.
  y-axis: top positive SSEI on any registered policy (model's largest credible effect).

A clean diagonal pattern - higher SSEI on the left, near-zero on the right -
is the visual restatement of the architecture-dependent finding the paper
foregrounds. Models with non-standard cache handling (gpt-oss MoE + harmony)
are flagged separately rather than placed on the scope axis.

Run: ``uv run python scripts/make_architecture_summary_figure.py``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Attention regime per panel model. 0 = vanilla full attention all layers,
# 1 = mixed local/global, 2 = full-layer sliding window, -1 = non-standard (MoE/harmony).
ATTENTION_REGIME = {
    "qwen2_5_7b_base": 0,
    "qwen2_5_7b_instruct": 0,
    "qwen3_5_9b": 0,
    "llama3_1_8b_instruct": 0,
    "olmo3_7b_instruct": 0,
    "phi4": 0,
    "gemma2_9b_it": 1,
    "mistral_7b_instruct_v0_3": 2,
    "gpt_oss_20b": -1,
    "qwen2_5_14b_msm_rules": 0,
    "qwen2_5_14b_instruct": 0,
}
REGIME_LABEL = {
    0: "Full global\nattention",
    1: "Mixed local\n+ global",
    2: "Sliding window\nevery layer",
}
MODEL_LABEL = {
    "qwen2_5_7b_base": "Qwen2.5-7B base",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen3_5_9b": "Qwen3-9B",
    "llama3_1_8b_instruct": "Llama-3.1-8B",
    "olmo3_7b_instruct": "OLMo-3-7B",
    "phi4": "Phi-4",
    "gemma2_9b_it": "Gemma-2-9B",
    "mistral_7b_instruct_v0_3": "Mistral-7B-v0.3",
    "gpt_oss_20b": "gpt-oss-20b",
    "qwen2_5_14b_msm_rules": "Qwen2.5-14B + MSM",
    "qwen2_5_14b_instruct": "Qwen2.5-14B-Instruct",
}


def collect_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(results_root.glob("selectivity_h200_powered_*")):
        if run_dir.name == "selectivity_h200_powered_combined":
            continue
        model_key = run_dir.name.removeprefix("selectivity_h200_powered_")
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        contrasts = metrics.get("policy_level_contrasts") or {}
        best_ssei = None
        best_lo = None
        for payload in contrasts.values():
            ssei = payload.get("selective_safety_erasure_index")
            ci = payload.get("selective_safety_erasure_index_ci") or {}
            if ssei is None:
                continue
            if best_ssei is None or ssei > best_ssei:
                best_ssei = ssei
                best_lo = ci.get("ci_low")
        if best_ssei is None:
            continue
        rows.append(
            {
                "model_key": model_key,
                "label": MODEL_LABEL.get(model_key, model_key),
                "regime": ATTENTION_REGIME.get(model_key, 0),
                "top_ssei": best_ssei,
                "ci_low": best_lo,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated/cross_model_visuals/16_architecture_summary.png"),
    )
    args = parser.parse_args()

    rows = collect_rows(args.results_root)
    if not rows:
        raise SystemExit("No selectivity panel runs found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6))

    # Standard-attention models on the regime axis
    rng = np.random.default_rng(42)
    drawn_at = {}
    for r in rows:
        if r["regime"] == -1:
            continue
        x = r["regime"] + rng.uniform(-0.15, 0.15)
        # Track horizontal positions used to avoid label collision
        drawn_at.setdefault(r["regime"], []).append((x, r["top_ssei"]))
        y = r["top_ssei"]
        positive = (r["ci_low"] or -1) > 0
        color = "#1d4ed8" if positive else "#9ca3af"
        ax.scatter(x, y, s=120, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(
            r["label"],
            (x, y),
            fontsize=9,
            xytext=(8, 6),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

    # gpt-oss-20b plotted off the scope axis as the separate failure mode
    for r in rows:
        if r["regime"] != -1:
            continue
        x = 3.0
        y = r["top_ssei"]
        ax.scatter(x, y, s=140, color="#dc2626", marker="X", edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(
            r["label"] + "\n(harmony-collapse\nfailure mode)",
            (x, y),
            fontsize=9,
            xytext=(8, -10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

    ax.axhline(0, color="#1f2937", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.01, color="#9ca3af", linewidth=0.5, linestyle=":", zorder=1)
    ax.text(3.5, 0.012, "SSEI threshold = 0.01", fontsize=8, color="#6b7280", ha="right")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([REGIME_LABEL[0], REGIME_LABEL[1], REGIME_LABEL[2], "Non-standard\ncache handling"], fontsize=9.5)
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(-0.1, 0.12)
    ax.set_ylabel("Top positive SSEI across registered policies", fontsize=10)
    ax.set_title(
        "Selective safety erasure scales with attention-scope architecture\n"
        "Blue = positive SSEI excluding 0; gray = sub-threshold or CI overlaps 0; red X = separate failure mode",
        fontsize=11,
    )
    ax.grid(axis="y", color="#f1f5f9", zorder=0)

    plt.tight_layout()
    fig.savefig(args.output, format="png", dpi=160)
    plt.close(fig)
    print(f"Wrote {args.output} ({len(rows)} models).")


if __name__ == "__main__":
    main()
