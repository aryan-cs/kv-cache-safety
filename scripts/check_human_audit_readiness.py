from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256

DEFAULT_REQUIRED_LABELS = [
    "human_refusal_correct",
    "human_unsafe_compliance",
    "human_system_leakage",
]
REQUIRED_RESULT_SOURCES = ["manifest.json", "generations.jsonl", "metrics.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check completed human audit quality gates.")
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--min-completion-rate", type=float, default=1.0)
    parser.add_argument("--min-label-n", type=int, default=1)
    parser.add_argument("--min-multi-annotator-fraction", type=float, default=1.0)
    parser.add_argument("--required-label", action="append", default=[])
    parser.add_argument("--require-baseline-deltas", action="store_true")
    parser.add_argument("--require-result-source-match", action="store_true")
    parser.add_argument("--allow-single-annotator", action="store_true")
    args = parser.parse_args()

    metrics = json.loads(args.summary_json.read_text(encoding="utf-8"))
    required_labels = args.required_label or DEFAULT_REQUIRED_LABELS
    failures = check_human_audit_readiness(
        metrics,
        min_completion_rate=args.min_completion_rate,
        min_label_n=args.min_label_n,
        min_multi_annotator_fraction=args.min_multi_annotator_fraction,
        required_labels=required_labels,
        require_baseline_deltas=args.require_baseline_deltas,
        allow_single_annotator=args.allow_single_annotator,
    )
    if args.require_result_source_match:
        if args.audit_manifest is None:
            failures.append("--require-result-source-match needs --audit-manifest")
        elif args.results_dir is None:
            failures.append("--require-result-source-match needs --results-dir")
        else:
            audit_manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
            failures.extend(check_audit_input_source_match(audit_manifest))
            failures.extend(check_audit_result_source_match(audit_manifest, args.results_dir))
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
    min_multi_annotator_fraction: float = 1.0,
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
    duplicate_keys = metrics.get("duplicate_annotation_keys") or []
    if duplicate_keys:
        failures.append(f"duplicate audit_id/annotator_id annotations present: {duplicate_keys[:5]}")

    label_rates = metrics.get("label_rates") or {}
    consensus_ties = metrics.get("label_consensus_ties") or {}
    for label in required_labels:
        label_rate = label_rates.get(label)
        if not isinstance(label_rate, dict):
            failures.append(f"missing label rate for `{label}`")
            continue
        n = int(label_rate.get("n") or 0)
        if n < min_label_n:
            failures.append(f"`{label}` has n={n}; need >= {min_label_n}")
        tied_ids = consensus_ties.get(label) or []
        if tied_ids:
            failures.append(f"`{label}` has unresolved consensus ties: {tied_ids[:5]}")
    label_context = metrics.get("label_context") or {}
    if "human_system_leakage" in required_labels:
        leakage_context = label_context.get("human_system_leakage") or {}
        missing_reference_count = int(leakage_context.get("missing_reference_count") or 0)
        if missing_reference_count:
            failures.append(
                "`human_system_leakage` has labels without required hidden/system "
                f"reference context: {missing_reference_count}"
            )

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
        distinct_annotator_count = int(metrics.get("distinct_annotator_count") or 0)
        if distinct_annotator_count < 2:
            failures.append(
                f"distinct_annotator_count={distinct_annotator_count}; need >= 2"
            )
        expected_audit_count = int(metrics.get("expected_audit_count") or 0)
        multi_annotator_count = int(metrics.get("multi_annotator_audit_count") or 0)
        required_multi = min_multi_annotator_fraction * expected_audit_count
        if expected_audit_count and multi_annotator_count < required_multi:
            failures.append(
                "multi_annotator_audit_fraction="
                f"{multi_annotator_count / expected_audit_count:.3f}; need >= "
                f"{min_multi_annotator_fraction:.3f}"
            )
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


def check_audit_result_source_match(
    audit_manifest: dict[str, Any],
    results_dir: Path,
    *,
    required_sources: list[str] | None = None,
) -> list[str]:
    failures: list[str] = []
    sources = required_sources or REQUIRED_RESULT_SOURCES
    result_sources = (audit_manifest.get("source_artifacts") or {}).get("results")
    if not isinstance(result_sources, dict):
        return ["audit manifest lacks result source artifacts; re-aggregate with --results-dir"]
    for name in sources:
        source = result_sources.get(name)
        if not isinstance(source, dict):
            failures.append(f"audit manifest lacks result source `{name}`")
            continue
        path = results_dir / name
        if not path.exists():
            failures.append(f"result source `{name}` is missing from {results_dir}")
            continue
        expected_sha = source.get("sha256")
        if not expected_sha:
            failures.append(f"audit manifest result source `{name}` lacks sha256")
            continue
        actual_sha = file_sha256(path)
        if expected_sha != actual_sha:
            failures.append(f"audit manifest result source `{name}` hash is stale")
    return failures


def check_audit_input_source_match(audit_manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    source_artifacts = audit_manifest.get("source_artifacts") or {}
    audit_csv_sources = source_artifacts.get("audit_csv")
    if not isinstance(audit_csv_sources, list) or not audit_csv_sources:
        failures.append("audit manifest lacks audit CSV source artifacts")
    else:
        for idx, source in enumerate(audit_csv_sources):
            failures.extend(_source_file_failures(source, f"audit CSV source {idx}"))
    key_source = source_artifacts.get("key_jsonl")
    if not isinstance(key_source, dict):
        failures.append("audit manifest lacks key JSONL source artifact")
    else:
        failures.extend(_source_file_failures(key_source, "key JSONL source"))
    export_manifest_source = source_artifacts.get("export_manifest")
    if not isinstance(export_manifest_source, dict):
        failures.append("audit manifest lacks audit export manifest source artifact")
    else:
        failures.extend(_source_file_failures(export_manifest_source, "audit export manifest source"))
    return failures


def _source_file_failures(source: Any, label: str) -> list[str]:
    if not isinstance(source, dict):
        return [f"{label} is malformed"]
    raw_path = source.get("path")
    if not raw_path:
        return [f"{label} lacks path"]
    path = Path(str(raw_path))
    if not path.exists():
        return [f"{label} is missing: {path}"]
    expected_sha = source.get("sha256")
    if not expected_sha:
        return [f"{label} lacks sha256"]
    if expected_sha != file_sha256(path):
        return [f"{label} hash is stale: {path}"]
    return []


if __name__ == "__main__":
    main()
