from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a stratified human-audit sheet from generations.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/audit"))
    parser.add_argument("--per-suite-policy", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.results_dir / "generations.jsonl")
    if not rows:
        raise SystemExit(f"No generations found in {args.results_dir}")
    run_id = args.results_dir.name
    sample = _stratified_sample(rows, args.per_suite_policy, args.seed)
    audit_rows = [_audit_row(row, run_id) for row in sample]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{run_id}_audit_sample.jsonl"
    csv_path = args.output_dir / f"{run_id}_audit_sample.csv"
    write_jsonl(jsonl_path, audit_rows)
    _write_csv(csv_path, audit_rows)
    print(f"Wrote {len(audit_rows)} audit rows to {csv_path} and {jsonl_path}")


def _stratified_sample(rows: list[dict[str, Any]], per_suite_policy: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("suite")), str(row.get("policy"))), []).append(row)
    sample: list[dict[str, Any]] = []
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        sample.extend(candidates[:per_suite_policy])
    return sample


def _audit_row(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": row.get("model_id"),
        "suite": row.get("suite"),
        "policy": row.get("policy"),
        "prompt_id": row.get("prompt_id"),
        "seed": row.get("seed"),
        "category": row.get("category"),
        "should_refuse": row.get("should_refuse"),
        "user": row.get("user"),
        "generated_text": row.get("generated_text"),
        "refusal_expected_accuracy": row.get("refusal_expected_accuracy"),
        "exact_leakage": row.get("exact_leakage"),
        "rouge_l_leakage_recall": row.get("rouge_l_leakage_recall"),
        "generated_word_count": row.get("generated_word_count"),
        "human_refusal_correct": "",
        "human_unsafe_compliance": "",
        "human_system_leakage": "",
        "human_notes": "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
