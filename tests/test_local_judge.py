from cache_safety_erasure.judging.local_cli import (
    build_judge_prompt,
    extract_first_json_object,
    judgment_key,
    parse_judge_json,
)


def test_extract_first_json_object_handles_wrapped_output() -> None:
    raw = 'notes before\n```json\n{"should_refuse": true}\n```\nnotes after'
    assert extract_first_json_object(raw) == '{"should_refuse": true}'


def test_parse_judge_json_normalizes_labels() -> None:
    labels = parse_judge_json(
        """
        {"should_refuse":"yes","refusal_correct":"no","unsafe_compliance":false,
        "system_leakage":"false","over_refusal":null,"capability_correct":"n/a",
        "confidence":0.75,"notes":"ambiguous"}
        """
    )
    assert labels["should_refuse"] is True
    assert labels["refusal_correct"] is False
    assert labels["unsafe_compliance"] is False
    assert labels["system_leakage"] is False
    assert labels["over_refusal"] is None
    assert labels["capability_correct"] is None
    assert labels["confidence"] == 0.75


def test_build_judge_prompt_includes_blinded_row_fields() -> None:
    row = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "suite": "refusal_safety",
        "prompt_id": "p1",
        "policy": "none",
        "seed": 0,
        "system": "Policy text",
        "user": "Harmful request",
        "generated_text": "I cannot help.",
    }
    prompt = build_judge_prompt(row)
    assert "Harmful request" in prompt
    assert "I cannot help." in prompt
    assert "Required JSON keys" in prompt
    assert judgment_key(row) == "Qwen/Qwen2.5-7B-Instruct::refusal_safety::p1::none::0"
