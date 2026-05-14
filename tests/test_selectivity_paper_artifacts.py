"""Tests for the selectivity-panel paper-artifact scripts.

Covers:
- scripts/make_family_replication_table.py
- scripts/make_cross_model_summary.py
- scripts/make_selectivity_claim_assessment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from make_cross_model_summary import find_top_ssei
from make_family_replication_table import collect_rows, render_latex, render_markdown
from make_selectivity_claim_assessment import (
    assess_claims,
    evaluate_model,
    render_claim_table,
    render_interpretation,
    render_status_sentence,
)


def _seed_model_dir(root: Path, model_key: str, contrasts: dict) -> Path:
    run_dir = root / f"selectivity_h200_powered_{model_key}"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"policy_level_contrasts": contrasts}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_family_replication_collects_only_models_with_contrasts(tmp_path: Path) -> None:
    _seed_model_dir(
        tmp_path,
        "mistral_7b_instruct_v0_3",
        {
            "sliding_window__budget128": {
                "selective_safety_erasure_index": 0.012,
                "selective_safety_erasure_index_ci": {"ci_low": 0.005, "ci_high": 0.02},
            }
        },
    )
    _seed_model_dir(tmp_path, "phi4_no_contrasts", {})

    rows = collect_rows(tmp_path)

    assert [r["model_key"] for r in rows] == ["mistral_7b_instruct_v0_3"]
    assert rows[0]["policy_level"]["sliding_window__budget128"]["ssei"] == 0.012


def test_family_replication_render_markdown_and_latex_have_one_row_per_model(tmp_path: Path) -> None:
    _seed_model_dir(
        tmp_path,
        "phi4",
        {
            "sliding_window__budget128": {
                "selective_safety_erasure_index": 0.084,
                "selective_safety_erasure_index_ci": {"ci_low": 0.076, "ci_high": 0.091},
            }
        },
    )

    rows = collect_rows(tmp_path)

    md = render_markdown(rows)
    assert md.count("|") > 0
    assert "Phi-4" in md and "0.084" in md

    tex = render_latex(rows)
    assert "\\begin{table}" in tex
    assert "Phi--4" in tex  # LaTeX renders hyphen as --
    assert "$0.084$" in tex


def test_cross_model_summary_find_top_ssei_returns_largest_positive() -> None:
    contrasts = {
        "policy_a": {
            "selective_safety_erasure_index": 0.005,
            "selective_safety_erasure_index_ci": {"ci_low": -0.001, "ci_high": 0.01},
        },
        "policy_b": {
            "selective_safety_erasure_index": 0.030,
            "selective_safety_erasure_index_ci": {"ci_low": 0.020, "ci_high": 0.040},
        },
        "policy_c": {
            "selective_safety_erasure_index": -0.010,
            "selective_safety_erasure_index_ci": {"ci_low": -0.02, "ci_high": 0.0},
        },
    }

    top_policy, top_ssei, lo, hi = find_top_ssei(contrasts)

    assert top_policy == "policy_b"
    assert top_ssei == 0.030
    assert lo == 0.020 and hi == 0.040


def test_evaluate_model_flags_positive_ci_excluding_zero() -> None:
    metrics = {
        "policy_level_contrasts": {
            "policy_a": {
                "selective_safety_erasure_index": 0.025,
                "selective_safety_erasure_index_ci": {"ci_low": 0.010, "ci_high": 0.040},
            },
            "policy_b": {
                "selective_safety_erasure_index": 0.030,
                "selective_safety_erasure_index_ci": {"ci_low": -0.005, "ci_high": 0.05},
            },
        }
    }
    audit_summary = {"input_rows": 200, "rows_with_any_parsed_judge": 195}

    summary = evaluate_model(metrics, audit_summary)

    assert summary["positive_with_ci_excluding_zero"] is True
    assert summary["best_ssei"] == 0.030
    assert summary["judging_coverage_rate"] == 195 / 200


def test_assess_claims_requires_two_families_for_cross_family_pass() -> None:
    rows = [
        {
            "model_key": "qwen_instruct",
            "family": "Qwen",
            "best_ssei": 0.02,
            "best_policy": "p_a",
            "best_ssei_ci_low": 0.005,
            "best_ssei_ci_high": 0.04,
            "positive_with_ci_excluding_zero": True,
            "judging_coverage_rate": 0.99,
            "judging_audit_rows": 200,
            "judging_parsed_rows": 198,
        },
    ]

    only_one_family = assess_claims(rows)

    assert only_one_family["claims"]["cross_family_replication"]["passed"] is False
    assert only_one_family["claims"]["safety_minus_capability_selectivity"]["passed"] is True


def test_render_status_sentence_renames_command_and_escapes_underscores() -> None:
    rows = [
        {
            "model_key": "mistral_7b_instruct_v0_3",
            "family": "Mistral",
            "best_ssei": 0.02,
            "best_policy": "p_a",
            "best_ssei_ci_low": 0.005,
            "best_ssei_ci_high": 0.04,
            "positive_with_ci_excluding_zero": True,
            "judging_coverage_rate": 1.0,
            "judging_audit_rows": 100,
            "judging_parsed_rows": 100,
        },
    ]
    assessment = assess_claims(rows)

    sentence = render_status_sentence(assessment)
    table = render_claim_table(assessment)
    interpretation = render_interpretation(assessment)

    assert "\\renewcommand{\\EmpiricalStatusSentence}" in sentence
    # underscores in the rendered prose must be LaTeX-escaped (preceded by \).
    body = sentence.split("\\renewcommand{\\EmpiricalStatusSentence}{%", 1)[-1]
    for idx, ch in enumerate(body):
        if ch == "_":
            assert idx > 0 and body[idx - 1] == "\\", body[max(0, idx - 5) : idx + 2]
    assert "\\begin{table}" in table and "\\bottomrule" in table
    assert "Claim interpretation" in interpretation
