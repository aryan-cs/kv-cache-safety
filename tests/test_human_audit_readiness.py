import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_human_audit_readiness import (
    check_audit_result_source_match,
    check_human_audit_readiness,
)


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


def test_audit_source_match_accepts_current_result_hashes(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_manifest = _audit_manifest_for(results_dir)

    failures = check_audit_result_source_match(audit_manifest, results_dir)

    assert failures == []


def test_audit_source_match_rejects_stale_or_missing_sources(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_manifest = _audit_manifest_for(results_dir)
    (results_dir / "metrics.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    del audit_manifest["source_artifacts"]["results"]["generations.jsonl"]

    failures = check_audit_result_source_match(audit_manifest, results_dir)

    assert "audit manifest lacks result source `generations.jsonl`" in failures
    assert "audit manifest result source `metrics.json` hash is stale" in failures


def _write_result_sources(results_dir: Path) -> Path:
    results_dir.mkdir()
    for name in ["manifest.json", "generations.jsonl", "metrics.json"]:
        (results_dir / name).write_text(f"{name}\n", encoding="utf-8")
    return results_dir


def _audit_manifest_for(results_dir: Path) -> dict:
    import hashlib

    return {
        "source_artifacts": {
            "results": {
                name: {"sha256": hashlib.sha256((results_dir / name).read_bytes()).hexdigest()}
                for name in ["manifest.json", "generations.jsonl", "metrics.json"]
            }
        }
    }
