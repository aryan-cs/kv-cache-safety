import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

import prepare_data
from prepare_data import load_hf_composite, write_suite_manifest

from cache_safety_erasure.evals.prompt_record import PromptRecord


def test_write_suite_manifest_hashes_processed_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "suite.jsonl"
    data_path.write_text('{"id":"p1"}\n', encoding="utf-8")

    manifest_path = write_suite_manifest(
        suite_name="suite",
        path=data_path,
        records=[PromptRecord(id="p1", suite="suite", user="hello")],
        source="builtin",
        source_args={"suite": "suite"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert len(manifest["sha256"]) == 64
    assert manifest["prompt_ids"] == ["p1"]


def test_load_hf_composite_fills_limit_from_multiple_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load_hf_preset(
        name: str, limit: int | None, output_suite: str | None, revision: str | None = None
    ) -> list[PromptRecord]:
        calls.append((name, limit, output_suite, revision))
        count = 2 if name == "advbench" else 3
        if limit is not None:
            count = min(count, limit)
        return [
            PromptRecord(id=f"{name}_{idx}", suite=str(output_suite), user=f"{name} prompt {idx}")
            for idx in range(count)
        ]

    monkeypatch.setattr(prepare_data, "load_hf_preset", fake_load_hf_preset)

    records = load_hf_composite("public_refusal_combo", 4, "public_refusal_safety")

    assert [record.id for record in records] == [
        "advbench_0",
        "advbench_1",
        "jailbreakbench_behaviors_0",
        "jailbreakbench_behaviors_1",
    ]
    assert {record.suite for record in records} == {"public_refusal_safety"}
    assert calls[0][0:3] == ("advbench", 4, "public_refusal_safety")
    assert calls[1][0:3] == ("jailbreakbench_behaviors", 2, "public_refusal_safety")
