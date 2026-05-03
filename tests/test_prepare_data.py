import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from prepare_data import write_suite_manifest

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
