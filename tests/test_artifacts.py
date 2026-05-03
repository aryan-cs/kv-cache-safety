from pathlib import Path

from cache_safety_erasure.utils.io import append_jsonl, read_jsonl, write_json


def test_json_artifacts_roundtrip(tmp_path: Path) -> None:
    write_json(tmp_path / "environment.json", {"ok": True})
    append_jsonl(tmp_path / "generations.jsonl", [{"prompt_id": "p1", "text": "hello"}])
    assert read_jsonl(tmp_path / "generations.jsonl")[0]["prompt_id"] == "p1"
