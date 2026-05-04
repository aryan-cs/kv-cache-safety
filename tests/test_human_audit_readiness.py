import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from aggregate_human_audit import aggregate_human_audit
from check_human_audit_readiness import (
    check_audit_input_source_match,
    check_audit_result_source_match,
    check_audit_summary_source_match,
    check_human_audit_readiness,
)


def _ready_metrics() -> dict:
    return {
        "expected_audit_count": 4,
        "completion_rate": 1.0,
        "unknown_audit_ids": [],
        "duplicate_annotation_keys": [],
        "distinct_annotator_count": 2,
        "multi_annotator_audit_count": 4,
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
        "label_consensus_ties": {},
        "label_context": {
            "human_system_leakage": {
                "n": 4,
                "missing_reference_count": 0,
                "missing_reference_audit_ids": [],
            }
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
    metrics["label_consensus_ties"] = {"human_refusal_correct": ["a1"]}
    metrics["multi_annotator_audit_count"] = 1
    metrics["label_context"]["human_system_leakage"]["missing_reference_count"] = 1

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
    assert any("consensus ties" in failure for failure in failures)
    assert any("multi_annotator_audit_fraction" in failure for failure in failures)
    assert any("reference context" in failure for failure in failures)


def test_human_audit_readiness_rejects_duplicate_or_single_annotator_audit() -> None:
    metrics = _ready_metrics()
    metrics["duplicate_annotation_keys"] = ["a1::r1"]
    metrics["distinct_annotator_count"] = 1

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

    assert any("duplicate audit_id/annotator_id" in failure for failure in failures)
    assert any("distinct_annotator_count" in failure for failure in failures)


def test_audit_source_match_accepts_current_result_hashes(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_manifest = _audit_manifest_for(results_dir, tmp_path / "audit")

    input_failures = check_audit_input_source_match(audit_manifest)
    failures = check_audit_result_source_match(audit_manifest, results_dir)

    assert input_failures == []
    assert failures == []


def test_audit_summary_source_match_accepts_recomputed_metrics(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_dir = tmp_path / "audit"
    audit_manifest = _audit_manifest_for(results_dir, audit_dir)
    labels = Path(audit_manifest["source_artifacts"]["audit_csv"][0]["path"])
    key = Path(audit_manifest["source_artifacts"]["key_jsonl"]["path"])
    summary = aggregate_human_audit([labels], key)["metrics"]

    failures = check_audit_summary_source_match(summary, audit_manifest)

    assert failures == []


def test_audit_summary_source_match_rejects_hand_edited_summary(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_dir = tmp_path / "audit"
    audit_manifest = _audit_manifest_for(results_dir, audit_dir)
    labels = Path(audit_manifest["source_artifacts"]["audit_csv"][0]["path"])
    key = Path(audit_manifest["source_artifacts"]["key_jsonl"]["path"])
    summary = aggregate_human_audit([labels], key)["metrics"]
    summary["completed_audit_count"] = 999
    summary["label_rates"]["human_refusal_correct"]["n"] = 999

    failures = check_audit_summary_source_match(summary, audit_manifest)

    assert any("completed_audit_count" in failure for failure in failures)
    assert any("label_rates.human_refusal_correct.n" in failure for failure in failures)


def test_audit_source_match_rejects_stale_or_missing_sources(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_manifest = _audit_manifest_for(results_dir, tmp_path / "audit")
    (results_dir / "metrics.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    del audit_manifest["source_artifacts"]["results"]["generations.jsonl"]

    failures = check_audit_result_source_match(audit_manifest, results_dir)

    assert "audit manifest lacks result source `generations.jsonl`" in failures
    assert "audit manifest result source `metrics.json` hash is stale" in failures


def test_audit_input_source_match_rejects_stale_or_missing_inputs(tmp_path: Path) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_dir = tmp_path / "audit"
    audit_manifest = _audit_manifest_for(results_dir, audit_dir)
    (audit_dir / "labels.csv").write_text("changed\n", encoding="utf-8")
    (audit_dir / "key.jsonl").unlink()

    failures = check_audit_input_source_match(audit_manifest)

    assert any("audit CSV source 0 hash is stale" in failure for failure in failures)
    assert any("key JSONL source is missing" in failure for failure in failures)


def _write_result_sources(results_dir: Path) -> Path:
    results_dir.mkdir()
    for name in ["manifest.json", "generations.jsonl", "metrics.json"]:
        (results_dir / name).write_text(f"{name}\n", encoding="utf-8")
    return results_dir


def _audit_manifest_for(results_dir: Path, audit_dir: Path) -> dict:
    import hashlib

    audit_dir.mkdir()
    labels = audit_dir / "labels.csv"
    key = audit_dir / "key.jsonl"
    export_manifest = audit_dir / "export_manifest.json"
    labels.write_text("audit_id,human_refusal_correct\n1,true\n", encoding="utf-8")
    key.write_text('{"audit_id":"1"}\n', encoding="utf-8")
    export_manifest.write_text(
        json.dumps(
            {
                "include_hidden_reference": True,
                "annotator_template_count": 2,
                "strategy": "effect",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    return {
        "source_artifacts": {
            "audit_csv": [
                {
                    "path": str(labels),
                    "sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                }
            ],
            "key_jsonl": {
                "path": str(key),
                "sha256": hashlib.sha256(key.read_bytes()).hexdigest(),
            },
            "export_manifest": {
                "path": str(export_manifest),
                "sha256": hashlib.sha256(export_manifest.read_bytes()).hexdigest(),
            },
            "results": {
                name: {"sha256": hashlib.sha256((results_dir / name).read_bytes()).hexdigest()}
                for name in ["manifest.json", "generations.jsonl", "metrics.json"]
            }
        }
    }


def test_audit_input_source_match_rejects_protocol_invalid_export_manifest(
    tmp_path: Path,
) -> None:
    results_dir = _write_result_sources(tmp_path / "results")
    audit_dir = tmp_path / "audit"
    audit_manifest = _audit_manifest_for(results_dir, audit_dir)
    export_manifest = Path(
        audit_manifest["source_artifacts"]["export_manifest"]["path"]
    )
    export_manifest.write_text(
        json.dumps(
            {
                "include_hidden_reference": False,
                "annotator_template_count": 1,
                "strategy": "",
            }
        ),
        encoding="utf-8",
    )
    audit_manifest["source_artifacts"]["export_manifest"]["sha256"] = _sha256(export_manifest)

    failures = check_audit_input_source_match(audit_manifest)

    assert "audit export manifest was not leakage-reference capable" in failures
    assert any("annotator_template_count=1" in failure for failure in failures)
    assert "audit export manifest lacks sampling strategy" in failures
    assert "audit export manifest lacks sampling seed" in failures


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
