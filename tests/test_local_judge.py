import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from approve_judge_egress import approve_rows
from judge_with_codex_gemini import _done_key, _judging_tasks

from cache_safety_erasure.judging.local_cli import (
    JudgeCommand,
    _judgment_record,
    build_judge_prompt,
    extract_first_json_object,
    infer_evaluated_family,
    infer_judge_family,
    judge_row,
    judgment_key,
    parse_judge_json,
    response_length_bucket,
    sha256_text,
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


def test_build_judge_prompt_blinds_protocol_fields_by_default() -> None:
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
    assert "Qwen/Qwen2.5-7B-Instruct" not in prompt
    assert "refusal_safety" not in prompt
    assert "policy: " not in prompt
    assert "none" not in prompt
    assert "seed" not in prompt
    assert judgment_key(row) == "Qwen/Qwen2.5-7B-Instruct::refusal_safety::p1::none::0"


def test_judgment_record_preserves_flat_and_nested_labels() -> None:
    row = {
        "model_id": "m",
        "suite": "s",
        "prompt_id": "p",
        "policy": "none",
        "seed": 0,
    }
    labels = {
        "should_refuse": True,
        "refusal_correct": False,
        "unsafe_compliance": True,
        "system_leakage": False,
        "over_refusal": False,
        "capability_correct": None,
        "confidence": 0.8,
        "notes": "ok",
    }

    record = _judgment_record(
        row,
        command=JudgeCommand(provider="codex", model="gpt-5.5"),
        prompt="prompt",
        raw_output="{}",
        started_at="2026-01-01T00:00:00Z",
        parser_status="parsed",
        labels=labels,
        command_line="codex exec",
        parse_error="",
        retry_count=1,
        judge_tool_version="codex-cli 1.2.3",
    )

    assert record["labels"] == labels
    assert record["unsafe_compliance"] is True
    assert record["raw_output_sha256"] == sha256_text("{}")
    assert record["parser_status"] == "parsed"
    assert record["retry_count"] == 1
    assert record["judge_tool_version"] == "codex-cli 1.2.3"
    assert record["judge_prompt_protocol_version"] == "blinded-v2"
    assert record["response_length_bucket"] == "empty"


def test_family_inference_blocks_codex_for_openai_family() -> None:
    row = {
        "model_id": "openai/gpt-oss-20b",
        "prompt_id": "p",
        "policy": "none",
        "data_egress_approved": True,
    }
    command = JudgeCommand(provider="codex", model="gpt-5.5")

    assert infer_evaluated_family(row) == "openai_gpt"
    assert infer_judge_family(command) == "openai_gpt"

    record = judge_row(row, [command], allow_data_egress=True)

    assert record["parser_status"] == "blocked"
    assert "matches judge family" in record["parse_error"]
    assert record["raw_output_sha256"] == sha256_text("")


def test_external_judging_requires_run_flag_and_row_approval() -> None:
    command = JudgeCommand(provider="gemini", model="gemini-3.1")
    approved_row = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_id": "p",
        "policy": "none",
        "data_egress_approved": True,
    }
    unapproved_row = {**approved_row, "data_egress_approved": False}

    missing_flag = judge_row(approved_row, [command], allow_data_egress=False)
    missing_row_approval = judge_row(unapproved_row, [command], allow_data_egress=True)

    assert missing_flag["parser_status"] == "blocked"
    assert "data egress flag not set" in missing_flag["parse_error"]
    assert missing_row_approval["parser_status"] == "blocked"
    assert "row lacks data_egress_approved=true" in missing_row_approval["parse_error"]


def test_approve_judge_egress_marks_rows_with_provenance() -> None:
    rows = [{"prompt_id": "p1", "generated_text": "answer"}]

    approved = approve_rows(
        rows,
        approval_note="approved for local model judging",
        approval_source="user_instruction",
        approved_at="2026-05-07T00:00:00Z",
    )

    assert approved[0]["prompt_id"] == "p1"
    assert approved[0]["data_egress_approved"] is True
    assert approved[0]["data_egress_approval_source"] == "user_instruction"
    assert approved[0]["data_egress_approval_note"] == "approved for local model judging"
    assert approved[0]["data_egress_approved_at"] == "2026-05-07T00:00:00Z"


def test_parse_failure_preserves_raw_output_hash_and_status(monkeypatch) -> None:
    raw_output = "not json"

    def fake_run(command: JudgeCommand, prompt: str) -> dict[str, str]:
        return {
            "raw_output": raw_output,
            "command_line": "gemini --prompt <prompt>",
            "tool_version": "gemini 3.1",
        }

    monkeypatch.setattr("cache_safety_erasure.judging.local_cli.run_judge_command", fake_run)
    row = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_id": "p",
        "policy": "none",
        "user": "Hello",
        "generated_text": "I cannot help with that request.",
        "data_egress_approved": True,
    }

    record = judge_row(
        row,
        [JudgeCommand(provider="gemini", model="gemini-3.1")],
        allow_data_egress=True,
    )

    assert record["parser_status"] == "parse_error"
    assert record["raw_output"] == raw_output
    assert record["raw_output_sha256"] == sha256_text(raw_output)
    assert record["labels"]["confidence"] == 0.0
    assert record["response_length_bucket"] == "short_1_50_words"


def test_response_length_bucket_boundaries() -> None:
    assert response_length_bucket("") == "empty"
    assert response_length_bucket("word " * 50) == "short_1_50_words"
    assert response_length_bucket("word " * 51) == "medium_51_200_words"


def test_judge_script_all_provider_tasks_preserve_disagreement_channels() -> None:
    row = {"model_id": "m", "suite": "s", "prompt_id": "p", "policy": "none", "seed": 0}
    commands = [
        JudgeCommand(provider="codex", model="gpt-5.5"),
        JudgeCommand(provider="gemini", model="gemini-3.1"),
    ]

    tasks = _judging_tasks([row], commands, set(), mode="all-providers")

    assert [task["commands"][0].provider for task in tasks] == ["codex", "gemini"]
    existing = {
        "judgment_key": judgment_key(row),
        "judge_provider": "codex",
        "judge_model": "gpt-5.5",
    }
    remaining = _judging_tasks([row], commands, {_done_key(existing, mode="all-providers")}, mode="all-providers")
    assert [task["commands"][0].provider for task in remaining] == ["gemini"]
