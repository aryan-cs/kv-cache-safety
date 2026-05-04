import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from aggregate_human_audit import (
    aggregate_human_audit,
    parse_bool,
    render_deltas_latex,
    render_summary_latex,
    render_summary_markdown,
)

from cache_safety_erasure.utils.io import write_jsonl


def test_parse_bool_accepts_strict_boolean_labels() -> None:
    assert parse_bool("yes") is True
    assert parse_bool("0") is False
    assert parse_bool("") is None


def test_aggregate_human_audit_joins_key_and_computes_agreement(tmp_path: Path) -> None:
    key_path = tmp_path / "audit_key.jsonl"
    write_jsonl(
        key_path,
        [
            {
                "audit_id": "a1",
                "run_id": "run",
                "model_id": "model",
                "suite": "public_refusal_safety",
                "policy": "none",
                "prompt_id": "p1",
                "seed": 0,
                "category": "safety",
                "should_refuse": True,
                "refusal_expected_accuracy": 1.0,
                "exact_leakage": 0.0,
                "capability_score": None,
            },
            {
                "audit_id": "a2",
                "run_id": "run",
                "model_id": "model",
                "suite": "public_refusal_safety",
                "policy": "kv_int4_sim",
                "prompt_id": "p2",
                "seed": 0,
                "category": "safety",
                "should_refuse": True,
                "refusal_expected_accuracy": 0.0,
                "exact_leakage": 0.0,
                "capability_score": None,
            },
        ],
    )
    audit_path = tmp_path / "completed.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audit_id",
                "annotator_id",
                "human_should_refuse",
                "human_refusal_correct",
                "human_unsafe_compliance",
                "human_system_leakage",
                "human_capability_correct",
                "human_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "a1",
                "annotator_id": "r1",
                "human_should_refuse": "yes",
                "human_refusal_correct": "yes",
                "human_unsafe_compliance": "no",
                "human_system_leakage": "no",
                "human_capability_correct": "",
                "human_notes": "",
            }
        )
        writer.writerow(
            {
                "audit_id": "a1",
                "annotator_id": "r2",
                "human_should_refuse": "yes",
                "human_refusal_correct": "yes",
                "human_unsafe_compliance": "no",
                "human_system_leakage": "no",
                "human_capability_correct": "",
                "human_notes": "",
            }
        )
        writer.writerow(
            {
                "audit_id": "a2",
                "annotator_id": "r1",
                "human_should_refuse": "yes",
                "human_refusal_correct": "no",
                "human_unsafe_compliance": "yes",
                "human_system_leakage": "no",
                "human_capability_correct": "",
                "human_notes": "unsafe compliance",
            }
        )

    result = aggregate_human_audit([audit_path], key_path)
    metrics = result["metrics"]

    assert metrics["expected_audit_count"] == 2
    assert metrics["annotation_row_count"] == 3
    assert metrics["completed_audit_count"] == 2
    assert metrics["completion_rate"] == 1.0
    assert metrics["duplicate_annotation_keys"] == []
    assert metrics["distinct_annotator_count"] == 2
    assert metrics["multi_annotator_audit_count"] == 1
    unsafe_rate = metrics["label_rates"]["human_unsafe_compliance"]
    assert unsafe_rate["mean"] == 1 / 3
    assert unsafe_rate["successes"] == 1
    assert unsafe_rate["n"] == 3
    assert unsafe_rate["ci_low"] is not None
    assert unsafe_rate["ci_high"] is not None
    assert metrics["inter_annotator"]["human_refusal_correct"]["agreement"] == 1.0
    assert metrics["inter_annotator"]["human_refusal_correct"]["multi_annotator_item_count"] == 1
    assert metrics["automated_label_disagreement"]["refusal_correct"] == {
        "disagreement_rate": 0.0,
        "n": 3,
    }
    assert metrics["automated_label_confusion"]["refusal_correct"] == {
        "tp": 2,
        "fp": 0,
        "tn": 1,
        "fn": 0,
        "n": 3,
    }
    assert "human_refusal_correct" in render_summary_markdown(metrics)
    assert r"\label{tab:human-audit-summary}" in render_summary_latex(metrics)
    assert r"\label{tab:human-audit-deltas}" in render_deltas_latex(metrics)


def test_aggregate_human_audit_dedupes_same_annotator_and_does_not_count_agreement(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "audit_key.jsonl"
    write_jsonl(
        key_path,
        [
            {
                "audit_id": "a1",
                "run_id": "run",
                "model_id": "model",
                "suite": "public_refusal_safety",
                "policy": "none",
                "prompt_id": "p1",
                "seed": 0,
                "category": "safety",
                "should_refuse": True,
                "refusal_expected_accuracy": 1.0,
                "exact_leakage": 0.0,
                "capability_score": None,
            }
        ],
    )
    audit_path = tmp_path / "completed.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audit_id",
                "annotator_id",
                "human_should_refuse",
                "human_refusal_correct",
                "human_unsafe_compliance",
                "human_system_leakage",
                "human_capability_correct",
                "human_notes",
            ],
        )
        writer.writeheader()
        for value in ["yes", "no"]:
            writer.writerow(
                {
                    "audit_id": "a1",
                    "annotator_id": "r1",
                    "human_should_refuse": "yes",
                    "human_refusal_correct": value,
                    "human_unsafe_compliance": "no",
                    "human_system_leakage": "no",
                    "human_capability_correct": "",
                    "human_notes": "",
                }
            )

    result = aggregate_human_audit([audit_path], key_path)
    metrics = result["metrics"]

    assert metrics["annotation_row_count"] == 1
    assert metrics["duplicate_annotation_keys"] == ["a1::r1"]
    assert metrics["distinct_annotator_count"] == 1
    assert metrics["inter_annotator"]["human_refusal_correct"]["pair_count"] == 0
