import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from sync_and_judge_selectivity_run import is_run_complete, judge_paths_for_run


def test_selectivity_judge_paths_stay_outside_h200_owned_results_dir(tmp_path: Path) -> None:
    paths = judge_paths_for_run("selectivity_run", tmp_path / "docs" / "audit")

    assert paths.audit_key == tmp_path / "docs" / "audit" / "selectivity_run_audit_key.jsonl"
    assert paths.approved_input.parent == tmp_path / "docs" / "audit"
    assert paths.judgments.parent == tmp_path / "docs" / "audit"
    assert "results" not in paths.judgments.parts


def test_is_run_complete_requires_expected_rows_and_complete_activity(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"expected_generation_count": 2}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "generations.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps({"expected": 2, "completed": 2, "activity": "generating"}) + "\n",
        encoding="utf-8",
    )

    complete, reason = is_run_complete(run_dir)

    assert complete is False
    assert "generating" in reason

    (run_dir / "progress.json").write_text(
        json.dumps({"expected": 2, "completed": 2, "activity": "complete"}) + "\n",
        encoding="utf-8",
    )

    complete, reason = is_run_complete(run_dir)

    assert complete is True
    assert reason == "2/2 rows complete"


def test_is_run_complete_reports_incomplete_row_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "progress.json").write_text(
        json.dumps({"expected": 2, "completed": 1, "activity": "generating"}) + "\n",
        encoding="utf-8",
    )

    complete, reason = is_run_complete(run_dir)

    assert complete is False
    assert reason == "1/2 rows complete"
