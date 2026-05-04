import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_prompt_disjointness import check_prompt_disjointness


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prompt_disjointness_passes_for_new_prompt_ids_and_text(tmp_path: Path) -> None:
    reference_dir = tmp_path / "results" / "primary"
    data_dir = tmp_path / "data" / "processed"
    _write_jsonl(
        reference_dir / "prompts.jsonl",
        [{"suite": "public_refusal_safety", "prompt_id": "p1", "user": "unsafe request one"}],
    )
    _write_jsonl(
        data_dir / "public_refusal_safety.jsonl",
        [{"suite": "public_refusal_safety", "id": "p2", "user": "unsafe request two"}],
    )

    failures = check_prompt_disjointness(
        reference_results_dir=reference_dir,
        data_dir=data_dir,
        suites=["public_refusal_safety"],
    )

    assert failures == []


def test_prompt_disjointness_fails_on_prompt_id_and_text_overlap(tmp_path: Path) -> None:
    reference_dir = tmp_path / "results" / "primary"
    data_dir = tmp_path / "data" / "processed"
    _write_jsonl(
        reference_dir / "generations.jsonl",
        [
            {
                "suite": "public_refusal_safety",
                "prompt_id": "p1",
                "policy": "none",
                "seed": 0,
                "user": "Unsafe   Request",
            }
        ],
    )
    _write_jsonl(
        data_dir / "public_refusal_safety.jsonl",
        [{"suite": "public_refusal_safety", "id": "p1", "user": "unsafe request"}],
    )

    failures = check_prompt_disjointness(
        reference_results_dir=reference_dir,
        data_dir=data_dir,
        suites=["public_refusal_safety"],
    )

    assert any("overlaps reference prompt IDs" in failure for failure in failures)
    assert any("overlaps exact normalized prompt text" in failure for failure in failures)
