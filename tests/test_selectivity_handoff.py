import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from write_selectivity_handoff import build_handoff


def test_selectivity_handoff_records_mac_judging_command(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "selectivity_h200_powered_qwen"
    run_dir.mkdir(parents=True)
    (run_dir / "progress.json").write_text(
        json.dumps({"activity": "complete", "completed": 2, "expected": 2}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_family": "Qwen 2.5",
                "model_track": "chat_safety",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "generations.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    handoff = build_handoff(
        run_dir=run_dir,
        config=Path("configs/experiments/selectivity_h200_powered_qwen.yaml"),
        stage="powered",
        model_key="qwen",
    )

    assert handoff["complete"] is True
    assert handoff["remote_run_dir"] == "results/selectivity_h200_powered_qwen"
    assert handoff["local_judging_command"] == [
        "uv",
        "run",
        "python",
        "scripts/sync_and_judge_selectivity_run.py",
        "--run-id",
        "selectivity_h200_powered_qwen",
        "--workers",
        "4",
    ]
    assert handoff["local_judging_outputs"]["judgments"] == (
        "docs/audit/selectivity_h200_powered_qwen_judgments.gemini.jsonl"
    )


def test_h200_panel_writes_handoff_after_each_run() -> None:
    script = Path("scripts/run_h200_selectivity_panel.sh").read_text(encoding="utf-8")

    assert "scripts/write_selectivity_handoff.py" in script
    assert "--model-key \"$key\"" in script
