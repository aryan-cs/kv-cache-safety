from __future__ import annotations

from pathlib import Path

from cache_safety_erasure.evals.prompt_record import PromptRecord
from cache_safety_erasure.evals.seed_suites import load_builtin_suite
from cache_safety_erasure.utils.io import read_jsonl, write_jsonl


def processed_suite_path(name: str, data_dir: Path = Path("data/processed")) -> Path:
    return data_dir / f"{name}.jsonl"


def write_prompt_suite(name: str, records: list[PromptRecord], data_dir: Path = Path("data/processed")) -> Path:
    path = processed_suite_path(name, data_dir)
    write_jsonl(path, [record.to_dict() for record in records])
    return path


def load_prompt_suite(name: str, data_dir: Path = Path("data/processed")) -> list[PromptRecord]:
    path = processed_suite_path(name, data_dir)
    if path.exists():
        return [PromptRecord.from_dict(row) for row in read_jsonl(path)]
    return load_builtin_suite(name)
