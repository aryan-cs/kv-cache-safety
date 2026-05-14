"""Smoke-test Claude Code (`claude -p`) as a judge against existing Gemini labels.

Picks a small random sample from a fully Gemini-judged model, asks Claude to
score each row using the same blinded rubric, and reports label agreement.
Intended as a single-shot pre-check before plumbing Claude in as a provider
in ``src/cache_safety_erasure/judging/local_cli.py``.

Run: ``uv run python scripts/calibrate_claude_vs_gemini.py``
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
sys.path.insert(0, str(Path("src").resolve()))

from cache_safety_erasure.judging.local_cli import (
    build_judge_prompt,
    extract_first_json_object,
    normalize_labels,
)

LABEL_FIELDS = (
    "should_refuse",
    "refusal_correct",
    "unsafe_compliance",
    "system_leakage",
    "over_refusal",
    "capability_correct",
)


def run_claude(prompt: str, model: str, timeout_seconds: int) -> dict:
    args = ["claude", "-p", "--output-format", "text", "--model", model]
    result = subprocess.run(
        args,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}: {result.stderr[:300]}")
    raw_output = result.stdout.strip()
    if not raw_output:
        raise RuntimeError(f"claude returned empty output; stderr: {result.stderr[:300]}")
    return {"raw_output": raw_output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-model",
        default="mistral_7b_instruct_v0_3",
        help="Model whose Gemini judgments are used as the comparison baseline.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/audit/claude_calibration_vs_gemini.json"),
    )
    args = parser.parse_args()

    key_path = Path(
        f"docs/audit/selectivity_h200_powered_{args.source_model}_audit_key.gemini_approved.jsonl"
    )
    judgments_path = Path(
        f"docs/audit/selectivity_h200_powered_{args.source_model}_judgments.gemini.jsonl"
    )
    if not key_path.exists() or not judgments_path.exists():
        raise SystemExit(f"Missing inputs: {key_path} or {judgments_path}")

    key_rows = {}
    with key_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("audit_id"):
                key_rows[r["audit_id"]] = r

    gemini_by_id = {}
    with judgments_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("parser_status") == "parsed" and r.get("audit_id"):
                gemini_by_id[r["audit_id"]] = r

    common_ids = sorted(set(key_rows) & set(gemini_by_id))
    if not common_ids:
        raise SystemExit("No overlap between key audit_ids and parsed Gemini judgments.")

    rng = random.Random(args.seed)
    sampled = rng.sample(common_ids, min(args.limit, len(common_ids)))
    print(f"Sampled {len(sampled)} audit_ids from {args.source_model}")

    per_field_agree = {f: 0 for f in LABEL_FIELDS}
    per_field_disagree = {f: 0 for f in LABEL_FIELDS}
    per_row_records = []
    started = time.time()
    for i, audit_id in enumerate(sampled, start=1):
        key_row = key_rows[audit_id]
        prompt = build_judge_prompt(key_row)
        try:
            result = run_claude(prompt, args.model, args.timeout_seconds)
            parsed_text = extract_first_json_object(result["raw_output"])
            claude_labels = normalize_labels(json.loads(parsed_text))
        except Exception as exc:  # noqa: BLE001
            per_row_records.append(
                {"audit_id": audit_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"  [{i}/{len(sampled)}] {audit_id}: ERROR {type(exc).__name__}")
            continue

        gemini_row = gemini_by_id[audit_id]
        gemini_labels = gemini_row.get("labels") or {f: gemini_row.get(f) for f in LABEL_FIELDS}
        gemini_norm = {f: gemini_labels.get(f) for f in LABEL_FIELDS}

        row_agreement = {}
        for f in LABEL_FIELDS:
            g_val = gemini_norm[f]
            c_val = claude_labels[f]
            agree = g_val == c_val
            row_agreement[f] = agree
            if agree:
                per_field_agree[f] += 1
            else:
                per_field_disagree[f] += 1
        per_row_records.append(
            {
                "audit_id": audit_id,
                "gemini": gemini_norm,
                "claude": {f: claude_labels[f] for f in LABEL_FIELDS},
                "agreement": row_agreement,
            }
        )
        print(
            f"  [{i}/{len(sampled)}] {audit_id}: "
            f"agree={sum(row_agreement.values())}/{len(LABEL_FIELDS)}"
        )

    elapsed = time.time() - started
    summary = {
        "source_model": args.source_model,
        "judge_model": args.model,
        "sampled": len(sampled),
        "elapsed_seconds": round(elapsed, 1),
        "errors": sum(1 for r in per_row_records if "error" in r),
        "per_field_agreement_rate": {
            f: (
                per_field_agree[f]
                / max(1, per_field_agree[f] + per_field_disagree[f])
            )
            for f in LABEL_FIELDS
        },
        "rows": per_row_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output_json}")
    print("Per-field agreement:")
    for f, rate in summary["per_field_agreement_rate"].items():
        print(f"  {f:>22s}: {rate * 100:5.1f}%")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / max(1, len(sampled)):.1f}s per row)")


if __name__ == "__main__":
    main()
