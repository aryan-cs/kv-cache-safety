from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.evals.io import processed_suite_path
from cache_safety_erasure.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if prepared prompt suites overlap a completed reference run. "
            "This protects CI-extension sweeps from silently rerunning the same prompt clusters."
        )
    )
    parser.add_argument("--reference-results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--suite", action="append", required=True)
    args = parser.parse_args()

    failures = check_prompt_disjointness(
        reference_results_dir=args.reference_results_dir,
        data_dir=args.data_dir,
        suites=args.suite,
    )
    if failures:
        print("PROMPT DISJOINTNESS CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PROMPT DISJOINTNESS CHECK PASSED")


def check_prompt_disjointness(
    *,
    reference_results_dir: Path,
    data_dir: Path,
    suites: list[str],
) -> list[str]:
    reference_rows = _load_reference_prompts(reference_results_dir)
    reference = _index_rows(reference_rows, suites)
    failures: list[str] = []
    if not reference_rows:
        failures.append(f"`{reference_results_dir}` has no prompts.jsonl or generations.jsonl rows")
        return failures

    for suite in suites:
        candidate_path = processed_suite_path(suite, data_dir)
        candidate_rows = read_jsonl(candidate_path)
        if not candidate_rows:
            failures.append(f"`{suite}` has no prepared candidate rows at {candidate_path}")
            continue
        candidate = _index_rows(candidate_rows, [suite])
        reference_suite = reference.get(suite, {"prompt_ids": set(), "user_hashes": set()})
        overlapping_prompt_ids = sorted(
            candidate[suite]["prompt_ids"].intersection(reference_suite["prompt_ids"])
        )
        overlapping_user_hashes = sorted(
            candidate[suite]["user_hashes"].intersection(reference_suite["user_hashes"])
        )
        if overlapping_prompt_ids:
            sample = ", ".join(overlapping_prompt_ids[:5])
            failures.append(
                f"`{suite}` overlaps reference prompt IDs: {len(overlapping_prompt_ids)} "
                f"overlap(s); first={sample}"
            )
        if overlapping_user_hashes:
            sample = ", ".join(overlapping_user_hashes[:5])
            failures.append(
                f"`{suite}` overlaps exact normalized prompt text: "
                f"{len(overlapping_user_hashes)} overlap(s); first_hash={sample}"
            )
    return failures


def _load_reference_prompts(reference_results_dir: Path) -> list[dict[str, Any]]:
    prompts_path = reference_results_dir / "prompts.jsonl"
    if prompts_path.exists():
        return read_jsonl(prompts_path)
    generations_path = reference_results_dir / "generations.jsonl"
    rows = read_jsonl(generations_path)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        suite = str(row.get("suite") or "")
        prompt_id = str(row.get("prompt_id") or "")
        if suite and prompt_id:
            unique.setdefault((suite, prompt_id), row)
    return list(unique.values())


def _index_rows(rows: list[dict[str, Any]], suites: list[str]) -> dict[str, dict[str, set[str]]]:
    suite_set = set(suites)
    index = {
        suite: {
            "prompt_ids": set(),
            "user_hashes": set(),
        }
        for suite in suites
    }
    for row in rows:
        suite = str(row.get("suite") or "")
        if suite not in suite_set:
            continue
        prompt_id = row.get("prompt_id", row.get("id"))
        if prompt_id not in {None, ""}:
            index[suite]["prompt_ids"].add(str(prompt_id))
        user = row.get("user")
        if isinstance(user, str) and user.strip():
            index[suite]["user_hashes"].add(_normalized_user_hash(user))
    return index


def _normalized_user_hash(user: str) -> str:
    normalized = " ".join(user.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
