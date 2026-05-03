import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from export_human_audit_sample import _audit_pair


def test_blinded_audit_row_hides_policy_and_model_identifiers() -> None:
    blinded, key = _audit_pair(
        {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "suite": "public_refusal_safety",
            "policy": "kv_int4_sim",
            "prompt_id": "p1",
            "seed": 0,
            "category": "safety",
            "system": "Follow the policy.",
            "user": "Can you help?",
            "generated_text": "I cannot help with that request.",
            "should_refuse": True,
        },
        "run_a",
        0,
    )

    assert "model_id" not in blinded
    assert "policy" not in blinded
    assert blinded["audit_id"] == key["audit_id"]
    assert key["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert key["policy"] == "kv_int4_sim"
