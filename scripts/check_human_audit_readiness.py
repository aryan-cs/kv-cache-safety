from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

DEFAULT_REQUIRED_LABELS = [
    "human_refusal_correct",
    "human_unsafe_compliance",
    "human_system_leakage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check completed human audit quality gates.")
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--min-completion-rate", type=float, default=1.0)
    parser.add_argument("--min-label-n", type=int, default=1)
    parser.add_argument("--required-label", action="append", default=[])
    parser.add_argument("--require-baseline-deltas", action="store_true")
    parser.add_argument("--allow-single-annotator", action="store_true")
    args = parser.parse_args()

    metrics = json.loads(args.summary_json.read_text(encoding="utf-8"))
    required_labels = args.required_label or DEFAULT_REQUIRED_LABELS
    failures = check_human_audit_readiness(
        metrics,
        min_completion_rate=args.min_completion_rate,
        min_label_n=args.min_label_n,
        required_labels=required_labels,
        require_baseline_deltas=args.require_baseline_deltas,
        allow_single_annotator=args.allow_single_annotator,
    )
    if failures:
        print("HUMAN AUDIT NOT READY")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("HUMAN AUDIT READY CHECK PASSED")


def check_human_audit_readiness(
    metrics: dict[str, Any],
    *,
    min_completion_rate: float,
    min_label_n: int,
    required_labels: list[str],
    require_baseline_deltas: bool,
    allow_single_annotator: bool,
) -> list[str]:
    failures: list[str] = []
    completion_rate = metrics.get("completion_rate")
    if completion_rate is None or float(completion_rate) < min_completion_rate:
        failures.append(
            f"completion_rate={completion_rate}; need >= {min_completion_rate:.3f}"
        )
    unknown_ids = metrics.get("unknown_audit_ids") or []
    if unknown_ids:
        failures.append(f"unknown audit ids present: {unknown_ids[:5]}")

    label_rates = metrics.get("label_rates") or {}
    for label in required_labels:
        label_rate = label_rates.get(label)
        if not isinstance(label_rate, dict):
            failures.append(f"missing label rate for `{label}`")
            continue
        n = int(label_rate.get("n") or 0)
        if n < min_label_n:
            failures.append(f"`{label}` has n={n}; need >= {min_label_n}")

    if require_baseline_deltas:
        deltas = metrics.get("baseline_policy_deltas") or {}
        for label in required_labels:
            matching = [
                values
                for key, values in deltas.items()
                if str(key).endswith(f"::{label}") and int(values.get("n") or 0) > 0
            ]
            if not matching:
                failures.append(f"missing paired baseline-policy deltas for `{label}`")

    if not allow_single_annotator:
        agreement = metrics.get("inter_annotator") or {}
        for label in required_labels:
            label_agreement = agreement.get(label)
            pair_count = (
                int(label_agreement.get("pair_count") or 0)
                if isinstance(label_agreement, dict)
                else 0
            )
            if pair_count < 1:
                failures.append(f"`{label}` has no inter-annotator pairs")
    return failures


if __name__ == "__main__":
    main()
