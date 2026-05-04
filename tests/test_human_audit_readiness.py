import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_human_audit_readiness import check_human_audit_readiness


def _ready_metrics() -> dict:
    return {
        "completion_rate": 1.0,
        "unknown_audit_ids": [],
        "label_rates": {
            "human_refusal_correct": {"n": 4},
            "human_unsafe_compliance": {"n": 4},
            "human_system_leakage": {"n": 4},
        },
        "baseline_policy_deltas": {
            "public_refusal_safety::kv_int4_sim::human_refusal_correct": {
                "treatment_minus_baseline": -0.25,
                "n": 2,
            },
            "public_refusal_safety::kv_int4_sim::human_unsafe_compliance": {
                "treatment_minus_baseline": 0.25,
                "n": 2,
            },
            "public_system_leakage::kv_int4_sim::human_system_leakage": {
                "treatment_minus_baseline": 0.25,
                "n": 2,
            },
        },
        "inter_annotator": {
            "human_refusal_correct": {"pair_count": 2},
            "human_unsafe_compliance": {"pair_count": 2},
            "human_system_leakage": {"pair_count": 2},
        },
    }


def test_human_audit_readiness_accepts_complete_paired_audit() -> None:
    failures = check_human_audit_readiness(
        _ready_metrics(),
        min_completion_rate=1.0,
        min_label_n=1,
        required_labels=[
            "human_refusal_correct",
            "human_unsafe_compliance",
            "human_system_leakage",
        ],
        require_baseline_deltas=True,
        allow_single_annotator=False,
    )

    assert failures == []


def test_human_audit_readiness_rejects_blank_or_unpaired_audit() -> None:
    metrics = _ready_metrics()
    metrics["completion_rate"] = 0.5
    metrics["label_rates"]["human_unsafe_compliance"]["n"] = 0
    metrics["baseline_policy_deltas"] = {}
    metrics["inter_annotator"]["human_system_leakage"]["pair_count"] = 0

    failures = check_human_audit_readiness(
        metrics,
        min_completion_rate=1.0,
        min_label_n=1,
        required_labels=[
            "human_refusal_correct",
            "human_unsafe_compliance",
            "human_system_leakage",
        ],
        require_baseline_deltas=True,
        allow_single_annotator=False,
    )

    assert any("completion_rate" in failure for failure in failures)
    assert any("human_unsafe_compliance" in failure for failure in failures)
    assert any("baseline-policy deltas" in failure for failure in failures)
    assert any("inter-annotator" in failure for failure in failures)
