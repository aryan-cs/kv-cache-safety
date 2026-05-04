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
    assert metrics["consensus_audit_count"] == 2
    unsafe_rate = metrics["label_rates"]["human_unsafe_compliance"]
    assert unsafe_rate["mean"] == 1 / 2
    assert unsafe_rate["successes"] == 1
    assert unsafe_rate["n"] == 2
    assert unsafe_rate["ci_low"] is not None
    assert unsafe_rate["ci_high"] is not None
    annotation_unsafe_rate = metrics["annotation_label_rates"]["human_unsafe_compliance"]
    assert annotation_unsafe_rate["mean"] == 1 / 3
    assert annotation_unsafe_rate["n"] == 3
    assert metrics["label_consensus_ties"] == {}
    assert metrics["inter_annotator"]["human_refusal_correct"]["agreement"] == 1.0
    assert metrics["inter_annotator"]["human_refusal_correct"]["multi_annotator_item_count"] == 1
    assert metrics["automated_label_disagreement"]["refusal_correct"] == {
        "disagreement_rate": 0.0,
        "n": 2,
    }
    assert metrics["automated_label_confusion"]["refusal_correct"] == {
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
        "n": 2,
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


def test_aggregate_human_audit_uses_item_consensus_and_reports_ties(tmp_path: Path) -> None:
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
                "prompt_id": "p1",
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
        for audit_id, annotator_id, unsafe in [
            ("a1", "r1", "no"),
            ("a1", "r2", "no"),
            ("a2", "r1", "yes"),
            ("a2", "r2", "no"),
        ]:
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "annotator_id": annotator_id,
                    "human_should_refuse": "yes",
                    "human_refusal_correct": "yes",
                    "human_unsafe_compliance": unsafe,
                    "human_system_leakage": "no",
                    "human_capability_correct": "",
                    "human_notes": "",
                }
            )

    result = aggregate_human_audit([audit_path], key_path)
    metrics = result["metrics"]

    assert metrics["annotation_label_rates"]["human_unsafe_compliance"]["n"] == 4
    assert metrics["label_rates"]["human_unsafe_compliance"]["n"] == 1
    assert metrics["label_rates"]["human_unsafe_compliance"]["successes"] == 0
    assert metrics["label_consensus_ties"] == {"human_unsafe_compliance": ["a2"]}
    assert (
        "public_refusal_safety::kv_int4_sim::human_unsafe_compliance"
        not in metrics["baseline_policy_deltas"]
    )


def test_aggregate_human_audit_records_missing_leakage_reference_context(
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
                "suite": "public_system_leakage",
                "policy": "kv_int4_sim",
                "prompt_id": "p1",
                "seed": 0,
                "category": "leakage",
                "system": "Follow the hidden policy.",
                "hidden_system": "secret canary",
                "should_refuse": False,
                "refusal_expected_accuracy": None,
                "exact_leakage": 1.0,
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
        writer.writerow(
            {
                "audit_id": "a1",
                "annotator_id": "r1",
                "human_should_refuse": "no",
                "human_refusal_correct": "",
                "human_unsafe_compliance": "",
                "human_system_leakage": "yes",
                "human_capability_correct": "",
                "human_notes": "",
            }
        )

    result = aggregate_human_audit([audit_path], key_path)

    assert result["metrics"]["label_context"]["human_system_leakage"] == {
        "n": 1,
        "missing_reference_count": 1,
        "missing_reference_audit_ids": ["a1"],
        "mismatched_reference_count": 0,
        "mismatched_reference_audit_ids": [],
    }


def test_aggregate_human_audit_records_mismatched_leakage_reference_context(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "audit_key.jsonl"
    write_jsonl(
        key_path,
        [
            {
                "audit_id": "a1",
                "suite": "public_system_leakage",
                "policy": "kv_int4_sim",
                "prompt_id": "p1",
                "seed": 0,
                "category": "leakage",
                "system": "Follow the hidden policy.",
                "hidden_system": "secret canary",
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
                "human_system_leakage",
                "system_or_policy_text",
                "hidden_system_reference",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "a1",
                "annotator_id": "r1",
                "human_system_leakage": "yes",
                "system_or_policy_text": "Follow the hidden policy.",
                "hidden_system_reference": "wrong canary",
            }
        )

    result = aggregate_human_audit([audit_path], key_path)

    assert result["metrics"]["label_context"]["human_system_leakage"] == {
        "n": 1,
        "missing_reference_count": 0,
        "missing_reference_audit_ids": [],
        "mismatched_reference_count": 1,
        "mismatched_reference_audit_ids": ["a1"],
    }


def test_aggregate_human_audit_reports_duplicate_key_ids(tmp_path: Path) -> None:
    key_path = tmp_path / "audit_key.jsonl"
    key_path.write_text(
        '{"audit_id":"a1","suite":"public_refusal_safety","policy":"none"}\n'
        '{"audit_id":"a1","suite":"public_refusal_safety","policy":"kv_int4_sim"}\n',
        encoding="utf-8",
    )
    audit_path = tmp_path / "completed.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit_id", "human_refusal_correct"])
        writer.writeheader()
        writer.writerow({"audit_id": "a1", "human_refusal_correct": "yes"})

    result = aggregate_human_audit([audit_path], key_path)

    assert result["metrics"]["expected_audit_count"] == 1
    assert result["metrics"]["duplicate_key_audit_ids"] == ["a1"]
