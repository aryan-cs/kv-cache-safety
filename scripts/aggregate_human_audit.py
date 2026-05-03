from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl, write_json

BOOLEAN_LABELS = [
    "human_should_refuse",
    "human_refusal_correct",
    "human_unsafe_compliance",
    "human_system_leakage",
    "human_capability_correct",
]
TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed human-audit annotations.")
    parser.add_argument("--audit-csv", required=True, nargs="+", type=Path)
    parser.add_argument("--key-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/audit"))
    args = parser.parse_args()

    result = aggregate_human_audit(args.audit_csv, args.key_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "human_audit_metrics.json", result["metrics"])
    _write_csv(args.output_dir / "human_audit_joined.csv", result["joined_rows"])
    (args.output_dir / "human_audit_summary.md").write_text(
        render_summary_markdown(result["metrics"]),
        encoding="utf-8",
    )
    print(f"Wrote human-audit aggregation to {args.output_dir}")


def aggregate_human_audit(audit_csv_paths: list[Path], key_jsonl_path: Path) -> dict[str, Any]:
    key_rows = {str(row["audit_id"]): row for row in read_jsonl(key_jsonl_path)}
    if not key_rows:
        raise ValueError(f"No key rows found in {key_jsonl_path}")
    annotation_rows = []
    for path in audit_csv_paths:
        annotation_rows.extend(_read_audit_csv(path))
    if not annotation_rows:
        raise ValueError("No audit annotations found.")

    joined_rows = []
    unknown_ids = []
    for row in annotation_rows:
        audit_id = str(row.get("audit_id", ""))
        key = key_rows.get(audit_id)
        if key is None:
            unknown_ids.append(audit_id)
            continue
        joined_rows.append(_joined_row(row, key))
    metrics = _audit_metrics(joined_rows, expected_audit_ids=set(key_rows), unknown_ids=unknown_ids)
    return {"metrics": metrics, "joined_rows": joined_rows}


def _read_audit_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _joined_row(annotation: dict[str, str], key: dict[str, Any]) -> dict[str, Any]:
    annotator_id = annotation.get("annotator_id") or annotation.get("rater_id") or "annotator_0"
    labels = {field: parse_bool(annotation.get(field)) for field in BOOLEAN_LABELS}
    return {
        "audit_id": key["audit_id"],
        "annotator_id": annotator_id,
        "run_id": key.get("run_id"),
        "model_id": key.get("model_id"),
        "suite": key.get("suite"),
        "policy": key.get("policy"),
        "prompt_id": key.get("prompt_id"),
        "seed": key.get("seed"),
        "category": key.get("category"),
        **labels,
        "human_notes": annotation.get("human_notes", ""),
        "auto_should_refuse": key.get("should_refuse"),
        "auto_refusal_expected_accuracy": key.get("refusal_expected_accuracy"),
        "auto_exact_leakage": key.get("exact_leakage"),
        "auto_rouge_l_leakage_recall": key.get("rouge_l_leakage_recall"),
        "auto_capability_score": key.get("capability_score"),
    }


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean label `{value}`. Use one of {sorted(TRUE_VALUES | FALSE_VALUES)} or blank."
    )


def _audit_metrics(
    joined_rows: list[dict[str, Any]],
    *,
    expected_audit_ids: set[str],
    unknown_ids: list[str],
) -> dict[str, Any]:
    completed_ids = {
        str(row["audit_id"])
        for row in joined_rows
        if any(row.get(field) is not None for field in BOOLEAN_LABELS)
    }
    by_suite_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        by_suite_policy[(str(row.get("suite")), str(row.get("policy")))].append(row)
    return {
        "schema_version": 1,
        "expected_audit_count": len(expected_audit_ids),
        "annotation_row_count": len(joined_rows),
        "completed_audit_count": len(completed_ids),
        "completion_rate": len(completed_ids) / len(expected_audit_ids) if expected_audit_ids else None,
        "unknown_audit_ids": sorted(set(unknown_ids)),
        "label_rates": {field: _rate(joined_rows, field) for field in BOOLEAN_LABELS},
        "automated_label_disagreement": _automated_disagreement(joined_rows),
        "inter_annotator": {
            field: _inter_annotator_agreement(joined_rows, field) for field in BOOLEAN_LABELS
        },
        "by_suite_policy": {
            f"{suite}::{policy}": {
                "count": len(rows),
                "label_rates": {field: _rate(rows, field) for field in BOOLEAN_LABELS},
                "automated_label_disagreement": _automated_disagreement(rows),
            }
            for (suite, policy), rows in sorted(by_suite_policy.items())
        },
    }


def _rate(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    if not values:
        return {"mean": None, "n": 0}
    return {"mean": float(mean(1.0 if value else 0.0 for value in values)), "n": len(values)}


def _automated_disagreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "should_refuse": (
            "human_should_refuse",
            "auto_should_refuse",
        ),
        "refusal_correct": (
            "human_refusal_correct",
            "auto_refusal_expected_accuracy",
        ),
        "system_leakage": (
            "human_system_leakage",
            "auto_exact_leakage",
        ),
        "capability_correct": (
            "human_capability_correct",
            "auto_capability_score",
        ),
    }
    output: dict[str, Any] = {}
    for name, (human_field, auto_field) in checks.items():
        pairs = []
        for row in rows:
            human_value = row.get(human_field)
            auto_value = _auto_bool(row.get(auto_field))
            if human_value is None or auto_value is None:
                continue
            pairs.append(human_value != auto_value)
        output[name] = {
            "disagreement_rate": float(mean(1.0 if value else 0.0 for value in pairs))
            if pairs
            else None,
            "n": len(pairs),
        }
    return output


def _auto_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return parse_bool(value)
    return bool(round(number))


def _inter_annotator_agreement(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_audit_id: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            by_audit_id[str(row["audit_id"])].append(bool(value))
    pairs = []
    for values in by_audit_id.values():
        if len(values) < 2:
            continue
        pairs.extend((left, right) for left, right in combinations(values, 2))
    if not pairs:
        return {"pair_count": 0, "agreement": None, "cohens_kappa": None}
    observed = mean(1.0 if left == right else 0.0 for left, right in pairs)
    left_true = mean(1.0 if left else 0.0 for left, _ in pairs)
    right_true = mean(1.0 if right else 0.0 for _, right in pairs)
    expected = left_true * right_true + (1.0 - left_true) * (1.0 - right_true)
    if abs(1.0 - expected) < 1e-12:
        kappa = None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "pair_count": len(pairs),
        "agreement": float(observed),
        "cohens_kappa": float(kappa) if kappa is not None else None,
    }


def render_summary_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Human Audit Summary",
        "",
        f"Expected audit items: `{metrics['expected_audit_count']}`",
        f"Completed audit items: `{metrics['completed_audit_count']}`",
        f"Completion rate: `{_format_float(metrics['completion_rate'])}`",
        "",
        "## Label Rates",
        "",
        "| label | mean | n |",
        "| --- | --- | --- |",
    ]
    for field, values in metrics["label_rates"].items():
        lines.append(f"| {field} | {_format_float(values['mean'])} | {values['n']} |")
    lines.extend(
        [
            "",
            "## Inter-Annotator Agreement",
            "",
            "| label | pair count | agreement | Cohen's kappa |",
            "| --- | --- | --- | --- |",
        ]
    )
    for field, values in metrics["inter_annotator"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    field,
                    str(values["pair_count"]),
                    _format_float(values["agreement"]),
                    _format_float(values["cohens_kappa"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
