import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from summarize_judge_attempts import summarize_judge_attempts


def test_summarize_judge_attempts_separates_raw_failures_from_repaired_coverage(tmp_path: Path) -> None:
    input_path = tmp_path / "approved.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"audit_id": "a1"}),
                json.dumps({"audit_id": "a2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    judgments_path.write_text(
        "\n".join(
            [
                json.dumps({"audit_id": "a1", "judge_provider": "gemini", "parser_status": "blocked"}),
                json.dumps({"audit_id": "a1", "judge_provider": "gemini", "parser_status": "parsed"}),
                json.dumps({"audit_id": "a2", "judge_provider": "gemini", "parser_status": "parsed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_judge_attempts(input_path, judgments_path)

    assert summary["input_rows"] == 2
    assert summary["judgment_attempt_rows"] == 3
    assert summary["provider_status_counts"] == {
        "gemini:blocked": 1,
        "gemini:parsed": 2,
    }
    assert summary["provider_coverage"]["gemini"] == {
        "attempted_rows": 2,
        "parsed_rows": 2,
        "missing_parsed_rows": 0,
    }
    assert summary["rows_with_any_parsed_judge"] == 2
    assert summary["rows_without_any_parsed_judge"] == 0
    assert summary["all_rows_have_parsed_judge"] is True
