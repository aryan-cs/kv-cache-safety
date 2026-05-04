import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from aggregate_human_audit import aggregate_human_audit, render_summary_latex
from run_open_judge_audit import parse_judge_json


def test_parse_open_judge_json_accepts_fenced_booleans() -> None:
    labels = parse_judge_json(
        """```json
        {
          "human_should_refuse": true,
          "human_refusal_correct": "no",
          "human_unsafe_compliance": "yes",
          "human_system_leakage": false,
          "human_capability_correct": null,
          "human_notes": "contains operational detail"
        }
        ```"""
    )

    assert labels["human_should_refuse"] == "yes"
    assert labels["human_refusal_correct"] == "no"
    assert labels["human_unsafe_compliance"] == "yes"
    assert labels["human_system_leakage"] == "no"
    assert labels["human_capability_correct"] == ""
    assert labels["human_notes"] == "contains operational detail"


def test_open_judge_audit_aggregation_is_not_rendered_as_human(tmp_path: Path) -> None:
    key = tmp_path / "audit_key.jsonl"
    key.write_text(
        "\n".join(
            [
                '{"audit_id":"a1","run_id":"r","model_id":"m","suite":"public_refusal_safety","policy":"none","prompt_id":"p1","seed":0,"category":"harmful"}',
                '{"audit_id":"a2","run_id":"r","model_id":"m","suite":"public_refusal_safety","policy":"kv_int4_sim","prompt_id":"p1","seed":0,"category":"harmful"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "open_judge.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audit_id",
                "annotator_id",
                "annotation_source_type",
                "open_judge_model_id",
                "open_judge_prompt_sha256",
                "human_should_refuse",
                "human_refusal_correct",
                "human_unsafe_compliance",
                "human_system_leakage",
                "human_capability_correct",
                "human_notes",
            ],
        )
        writer.writeheader()
        for audit_id, unsafe in [("a1", "no"), ("a2", "yes")]:
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "annotator_id": "open_judge_qwen_v1",
                    "annotation_source_type": "open_local_judge",
                    "open_judge_model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "open_judge_prompt_sha256": "abc",
                    "human_should_refuse": "yes",
                    "human_refusal_correct": "yes",
                    "human_unsafe_compliance": unsafe,
                    "human_system_leakage": "no",
                    "human_capability_correct": "",
                    "human_notes": "",
                }
            )

    result = aggregate_human_audit([csv_path], key)
    metrics = result["metrics"]

    assert metrics["annotation_source_type"] == "open_local_judge"
    assert metrics["annotation_source_description"].startswith("Open local judge labels")
    rendered = render_summary_latex(metrics)
    assert "open local judge audit label rates" in rendered
    assert "human-audit label rates" not in rendered
