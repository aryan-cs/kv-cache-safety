import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from assess_claims import assess_claims, render_latex_table


def test_claim_assessment_passes_only_with_causal_system_control_gap() -> None:
    assessment = assess_claims(_primary_positive_metrics(), _causal_positive_metrics())

    assert assessment["publication_gate"]["passed"] is True
    assert assessment["claims"]["H1_behavioral_cache_sensitivity"]["passed"] is True
    assert assessment["claims"]["H2_selective_safety_degradation"]["passed"] is True
    assert assessment["claims"]["H3_causal_safety_state_erasure"]["passed"] is True
    assert "cache-mediated safety erasure" in assessment["recommended_framing"]


def test_claim_assessment_rejects_selective_effect_without_causal_control_gap() -> None:
    causal = _causal_positive_metrics()
    causal["causal_restoration"][
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem"
    ]["safety_restoration_fraction"] = 0.60

    assessment = assess_claims(_primary_positive_metrics(), causal)

    assert assessment["claims"]["H1_behavioral_cache_sensitivity"]["passed"] is True
    assert assessment["claims"]["H2_selective_safety_degradation"]["passed"] is True
    assert assessment["claims"]["H3_causal_safety_state_erasure"]["passed"] is False
    assert assessment["publication_gate"]["passed"] is False
    assert "but not the causal" in assessment["recommended_framing"]


def test_claim_assessment_rejects_missing_intervals() -> None:
    assessment = assess_claims({"selective_safety_erasure": {}}, {"causal_restoration": {}})

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["claims"]["H1_behavioral_cache_sensitivity"]["passed"] is False
    assert "no eligible interval" in assessment["claims"]["H1_behavioral_cache_sensitivity"]["summary"]


def test_claim_assessment_latex_table_is_formal_and_escaped() -> None:
    assessment = assess_claims(_primary_positive_metrics(), _causal_positive_metrics())

    table = render_latex_table(assessment)

    assert r"\label{tab:claim-assessment}" in table
    assert r"95\% CI" in table
    assert "kv\\_int4\\_sim" in table


def test_claim_assessment_require_flag_failure(tmp_path: Path) -> None:
    import json
    import subprocess

    primary_dir = tmp_path / "primary"
    causal_dir = tmp_path / "causal"
    primary_dir.mkdir()
    causal_dir.mkdir()
    (primary_dir / "metrics.json").write_text(
        json.dumps({"selective_safety_erasure": {}}), encoding="utf-8"
    )
    (causal_dir / "metrics.json").write_text(
        json.dumps({"causal_restoration": {}}), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/assess_claims.py",
            "--primary-results-dir",
            str(primary_dir),
            "--causal-results-dir",
            str(causal_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--require-cache-mediated-claim",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "do not yet justify" in result.stderr


def _primary_positive_metrics() -> dict:
    return {
        "selective_safety_erasure": {
            "public_refusal_safety::kv_int4_sim": {
                "safety_degradation": 0.12,
                "paired_safety_degradation_ci": {
                    "ci_low": 0.05,
                    "ci_high": 0.18,
                    "paired_n": 100,
                    "cluster_n": 100,
                },
            }
        },
        "policy_level_contrasts": {
            "kv_int4_sim": {
                "selective_safety_erasure_index": 0.08,
                "selective_safety_erasure_index_ci": {
                    "mean": 0.08,
                    "ci_low": 0.03,
                    "ci_high": 0.14,
                    "n_safety": 100,
                    "n_capability": 100,
                },
            }
        },
    }


def _causal_positive_metrics() -> dict:
    return {
        "causal_restoration": {
            "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                "compressed_policy": "kv_int4_sim",
                "safety_restoration_fraction": 0.62,
                "refusal_restoration_fraction": 0.55,
            },
            "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem": {
                "compressed_policy": "kv_int4_sim",
                "safety_restoration_fraction": 0.20,
                "refusal_restoration_fraction": 0.18,
            },
        }
    }
