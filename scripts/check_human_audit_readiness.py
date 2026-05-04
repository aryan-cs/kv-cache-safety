from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from aggregate_human_audit import aggregate_human_audit

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
            failures.extend(check_audit_summary_source_match(metrics, audit_manifest))
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
        failures.extend(_export_manifest_protocol_failures(export_manifest_source))
    return failures


def check_audit_summary_source_match(
    summary: dict[str, Any], audit_manifest: dict[str, Any]
) -> list[str]:
    source_artifacts = audit_manifest.get("source_artifacts") or {}
    audit_csv_sources = source_artifacts.get("audit_csv")
    key_source = source_artifacts.get("key_jsonl")
    if not isinstance(audit_csv_sources, list) or not audit_csv_sources:
        return ["cannot recompute audit summary without audit CSV sources"]
    if not isinstance(key_source, dict) or not key_source.get("path"):
        return ["cannot recompute audit summary without key JSONL source"]
    audit_csv_paths = [Path(str(source.get("path", ""))) for source in audit_csv_sources]
    key_jsonl_path = Path(str(key_source["path"]))
    missing_inputs = [str(path) for path in [*audit_csv_paths, key_jsonl_path] if not path.exists()]
    if missing_inputs:
        return [
            "cannot recompute audit summary because source files are missing: "
            + ", ".join(missing_inputs[:5])
        ]
    try:
        recomputed = aggregate_human_audit(audit_csv_paths, key_jsonl_path)["metrics"]
    except Exception as exc:
        return [f"cannot recompute audit summary from raw annotations: {exc}"]

    failures: list[str] = []
    for key in [
        "schema_version",
        "expected_audit_count",
        "annotation_row_count",
        "completed_audit_count",
        "consensus_audit_count",
        "completion_rate",
        "distinct_annotator_count",
        "multi_annotator_audit_count",
    ]:
        failures.extend(_summary_value_mismatch(summary, recomputed, key))
    for key in ["unknown_audit_ids", "duplicate_annotation_keys", "label_consensus_ties"]:
        if _normalized_json(summary.get(key)) != _normalized_json(recomputed.get(key)):
            failures.append(f"audit summary `{key}` does not match raw annotation recomputation")
    for label in DEFAULT_REQUIRED_LABELS:
        summary_rate = (summary.get("label_rates") or {}).get(label)
        recomputed_rate = (recomputed.get("label_rates") or {}).get(label)
        if not isinstance(summary_rate, dict) or not isinstance(recomputed_rate, dict):
            failures.append(f"audit summary label rate `{label}` cannot be recomputed")
            continue
        for key in ["n", "successes", "mean", "ci_low", "ci_high"]:
            failures.extend(
                _summary_value_mismatch(
                    summary_rate,
                    recomputed_rate,
                    key,
                    label=f"label_rates.{label}.{key}",
                )
            )
    failures.extend(
        _baseline_delta_mismatches(
            summary.get("baseline_policy_deltas") or {},
            recomputed.get("baseline_policy_deltas") or {},
        )
    )
    return failures


def _baseline_delta_mismatches(summary: dict[str, Any], recomputed: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    summary_keys = set(str(key) for key in summary)
    recomputed_keys = set(str(key) for key in recomputed)
    for key in sorted(recomputed_keys - summary_keys):
        failures.append(f"audit summary missing recomputed baseline delta `{key}`")
    for key in sorted(summary_keys - recomputed_keys):
        failures.append(f"audit summary has non-recomputed baseline delta `{key}`")
    for key in sorted(summary_keys & recomputed_keys):
        summary_values = summary.get(key) or {}
        recomputed_values = recomputed.get(key) or {}
        if not isinstance(summary_values, dict) or not isinstance(recomputed_values, dict):
            failures.append(f"audit summary baseline delta `{key}` is malformed")
            continue
        for value_key in ["n", "treatment_minus_baseline"]:
            failures.extend(
                _summary_value_mismatch(
                    summary_values,
                    recomputed_values,
                    value_key,
                    label=f"baseline_policy_deltas.{key}.{value_key}",
                )
            )
    return failures


def _summary_value_mismatch(
    summary: dict[str, Any],
    recomputed: dict[str, Any],
    key: str,
    *,
    label: str | None = None,
) -> list[str]:
    summary_value = summary.get(key)
    recomputed_value = recomputed.get(key)
    if _values_match(summary_value, recomputed_value):
        return []
    name = label or key
    return [
        f"audit summary `{name}`={summary_value!r} does not match "
        f"raw annotation recomputation {recomputed_value!r}"
    ]


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            left_float = float(left)
            right_float = float(right)
        except (TypeError, ValueError):
            return False
        return math.isclose(left_float, right_float, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _normalized_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _export_manifest_protocol_failures(source: dict[str, Any]) -> list[str]:
    raw_path = source.get("path")
    if not raw_path:
        return []
    path = Path(str(raw_path))
    if not path.exists():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"audit export manifest is invalid JSON: {exc}"]
    failures = []
    if manifest.get("include_hidden_reference") is not True:
        failures.append("audit export manifest was not leakage-reference capable")
    annotator_template_count = int(manifest.get("annotator_template_count") or 0)
    if annotator_template_count < 2:
        failures.append(
            f"audit export manifest annotator_template_count={annotator_template_count}; need >= 2"
        )
    if not manifest.get("strategy"):
        failures.append("audit export manifest lacks sampling strategy")
    if "seed" not in manifest:
        failures.append("audit export manifest lacks sampling seed")
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
