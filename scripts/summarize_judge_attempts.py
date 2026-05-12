from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl_tolerant


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize append-only local judge attempts separately from repaired "
            "row/provider coverage. Model-judge labels remain source-marked and are not human labels."
        )
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--judgments-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_judge_attempts(args.input_jsonl, args.judgments_jsonl)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(text, end="")
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


def summarize_judge_attempts(input_jsonl: Path, judgments_jsonl: Path) -> dict[str, Any]:
    input_rows, corrupt_input = read_jsonl_tolerant(input_jsonl)
    judgment_rows, corrupt_judgments = read_jsonl_tolerant(judgments_jsonl)

    audit_ids = {str(row.get("audit_id")) for row in input_rows if row.get("audit_id")}
    provider_status_counts: Counter[str] = Counter()
    provider_attempt_ids: dict[str, set[str]] = defaultdict(set)
    provider_parsed_ids: dict[str, set[str]] = defaultdict(set)
    any_parsed_ids: set[str] = set()

    for row in judgment_rows:
        provider = str(row.get("judge_provider") or "unknown")
        status = str(row.get("parser_status") or "unknown")
        audit_id = str(row.get("audit_id") or "")
        provider_status_counts[f"{provider}:{status}"] += 1
        if audit_id:
            provider_attempt_ids[provider].add(audit_id)
            if status == "parsed":
                provider_parsed_ids[provider].add(audit_id)
                any_parsed_ids.add(audit_id)

    providers = sorted(provider_attempt_ids)
    provider_coverage = {
        provider: {
            "attempted_rows": len(provider_attempt_ids[provider] & audit_ids),
            "parsed_rows": len(provider_parsed_ids[provider] & audit_ids),
            "missing_parsed_rows": len(audit_ids - provider_parsed_ids[provider]),
        }
        for provider in providers
    }

    return {
        "input_rows": len(input_rows),
        "judgment_attempt_rows": len(judgment_rows),
        "corrupt_input_tail": str(corrupt_input) if corrupt_input is not None else "",
        "corrupt_judgment_tail": str(corrupt_judgments) if corrupt_judgments is not None else "",
        "provider_status_counts": dict(sorted(provider_status_counts.items())),
        "provider_coverage": provider_coverage,
        "rows_with_any_parsed_judge": len(any_parsed_ids & audit_ids),
        "rows_without_any_parsed_judge": len(audit_ids - any_parsed_ids),
        "all_rows_have_parsed_judge": audit_ids <= any_parsed_ids,
    }


if __name__ == "__main__":
    main()
