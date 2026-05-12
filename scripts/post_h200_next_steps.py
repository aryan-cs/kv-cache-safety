from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from report_publication_status import publication_status

from cache_safety_erasure.utils.io import write_json

_DERIVED_RESULT_ARTIFACTS = {"metrics.json", "figures/manifest.json"}
_DERIVED_READINESS_EXACT_FAILURES = {
    "missing generated PNG figures",
    "missing figures/manifest.json",
    "figures manifest has no figure entries",
    "figures manifest contains non-object entry",
    "causal patch runs require causal_restoration_fraction figure",
}
_DERIVED_READINESS_PREFIXES = (
    "figure `",
    "figures manifest ",
    "figures_manifest_",
    "invalid figures/manifest.json:",
    "missing required figure `",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the next fail-closed step after H200 result generation."
    )
    parser.add_argument(
        "--primary-results-dir",
        type=Path,
        default=Path("results/h200_qwen_full_sweep"),
    )
    parser.add_argument(
        "--causal-results-dir",
        type=Path,
        default=Path("results/h200_causal_patch_qwen7b"),
    )
    parser.add_argument(
        "--primary-audit-dir",
        type=Path,
        default=Path("docs/audit/h200_qwen_full_sweep_summary"),
    )
    parser.add_argument(
        "--causal-audit-dir",
        type=Path,
        default=Path("docs/audit/h200_causal_patch_qwen7b_summary"),
    )
    parser.add_argument(
        "--primary-generated-dir",
        type=Path,
        default=None,
        help="Generated paper asset directory for the selected primary result set.",
    )
    parser.add_argument(
        "--causal-generated-dir",
        type=Path,
        default=None,
        help="Generated paper asset directory for the selected causal result set.",
    )
    parser.add_argument(
        "--claim-assessment",
        type=Path,
        default=Path("docs/generated/claim_assessment/claim_assessment.json"),
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=Path("docs/kv-cache-safety.pdf"),
    )
    parser.add_argument(
        "--arxiv-source-dir",
        type=Path,
        default=Path("docs/build/arxiv_source"),
    )
    parser.add_argument(
        "--arxiv-archive",
        type=Path,
        default=Path("docs/build/arxiv_source.tar.gz"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args()

    primary_generated_dir = (
        args.primary_generated_dir or Path("docs/generated") / args.primary_results_dir.name
    )
    causal_generated_dir = (
        args.causal_generated_dir or Path("docs/generated") / args.causal_results_dir.name
    )
    status = publication_status(
        primary_results_dir=args.primary_results_dir,
        causal_results_dir=args.causal_results_dir,
        primary_audit_dir=args.primary_audit_dir,
        causal_audit_dir=args.causal_audit_dir,
        claim_assessment_path=args.claim_assessment,
        primary_generated_dir=primary_generated_dir,
        causal_generated_dir=causal_generated_dir,
        paper_pdf=args.paper_pdf,
        arxiv_source_dir=args.arxiv_source_dir,
        arxiv_archive=args.arxiv_archive,
        require_arxiv_bundle=True,
    )
    status["primary_generated"] = {"path": str(primary_generated_dir)}
    status["causal_generated"] = {"path": str(causal_generated_dir)}
    report = post_h200_next_steps(status)
    if args.output_json is not None:
        write_json(args.output_json, report)
    markdown = render_markdown(report)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.fail_if_not_ready and not report["publication_ready"]:
        raise SystemExit(1)


def post_h200_next_steps(status: dict[str, Any]) -> dict[str, Any]:
    gates = status.get("gates") or {}
    primary_results_path = _status_path(
        status,
        "primary_results",
        "results/h200_qwen_full_sweep",
    )
    causal_results_path = _status_path(
        status,
        "causal_results",
        "results/h200_causal_patch_qwen7b",
    )
    primary_audit_path = _status_path(
        status,
        "primary_human_audit",
        f"docs/audit/{Path(primary_results_path).name}_summary",
    )
    causal_audit_path = _status_path(
        status,
        "causal_human_audit",
        f"docs/audit/{Path(causal_results_path).name}_summary",
    )
    claim_assessment_path = _status_path(
        status,
        "claim_assessment",
        "docs/generated/claim_assessment/claim_assessment.json",
    )
    primary_generated_path = _status_path(
        status,
        "primary_generated",
        f"docs/generated/{Path(primary_results_path).name}",
    )
    causal_generated_path = _status_path(
        status,
        "causal_generated",
        f"docs/generated/{Path(causal_results_path).name}",
    )
    arxiv_source_path = _artifact_field(
        status,
        "arxiv_bundle",
        "source_dir",
        "docs/build/arxiv_source",
    )
    arxiv_archive_path = _artifact_field(
        status,
        "arxiv_bundle",
        "archive",
        "docs/build/arxiv_source.tar.gz",
    )
    claim_generated_path = str(Path(claim_assessment_path).parent)
    primary_raw_complete = bool(gates.get("primary_results_complete")) or _raw_result_available(
        status.get("primary_results")
    )
    causal_raw_complete = bool(gates.get("causal_results_complete")) or _raw_result_available(
        status.get("causal_results")
    )
    primary_evidence_prepared = bool(gates.get("primary_results_complete")) or _prepared_result_available(
        status.get("primary_results")
    )
    causal_evidence_prepared = bool(gates.get("causal_results_complete")) or _prepared_result_available(
        status.get("causal_results")
    )
    fetched_evidence_prepared = primary_evidence_prepared and causal_evidence_prepared
    ci_width_blocked = fetched_evidence_prepared and _has_ci_width_blockers(status)
    steps = [
        _step(
            "complete_h200_results",
            complete=primary_raw_complete and causal_raw_complete,
            ready=True,
            command="bash scripts/wait_and_run_h200_sweep.sh",
            detail="Run or wait for the registered H200 launcher until primary and causal raw result directories are ready to fetch.",
        ),
        _step(
            "prepare_after_h200_fetch",
            complete=fetched_evidence_prepared,
            ready=primary_raw_complete and causal_raw_complete,
            command=(
                "bash scripts/fetch_h200_results.sh "
                f"{_q(primary_results_path)} "
                f"{_q(causal_results_path)} && "
                f"PRIMARY_RESULTS_DIR={_q(primary_results_path)} "
                f"CAUSAL_RESULTS_DIR={_q(causal_results_path)} "
                f"PRIMARY_GENERATED_DIR={_q(primary_generated_path)} "
                f"CAUSAL_GENERATED_DIR={_q(causal_generated_path)} "
                f"PRIMARY_AUDIT_SUMMARY_DIR={_q(primary_audit_path)} "
                f"CAUSAL_AUDIT_SUMMARY_DIR={_q(causal_audit_path)} "
                f"ARXIV_SOURCE_DIR={_q(arxiv_source_path)} "
                f"ARXIV_ARCHIVE={_q(arxiv_archive_path)} "
                "bash scripts/prepare_after_h200_fetch.sh"
            ),
            detail="Fetch raw H200 evidence, then reaggregate metrics, regenerate figures and paper tables, run readiness checks, and export audit templates from the current clean local checkout.",
        ),
        _step(
            "resolve_ci_width",
            complete=not ci_width_blocked,
            ready=fetched_evidence_prepared and ci_width_blocked,
            command=(
                f"PRIMARY_RESULTS_DIR={_q(primary_results_path)} "
                "SWEEP_SCRIPT=scripts/run_h200_ci_extension.sh "
                "bash scripts/wait_and_run_h200_sweep.sh"
            ),
            detail=(
                "Run the registered CI extension through the guarded H200 launcher before "
                "publication if completed result artifacts are otherwise valid but confidence "
                "intervals exceed the configured width gate. Do not change thresholds or "
                "claim wording to bypass this gate."
            ),
        ),
        _step(
            "complete_human_audits",
            complete=bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete")),
            ready=fetched_evidence_prepared and not ci_width_blocked,
            command=(
                f"PRIMARY_RUN_ID={_q(Path(primary_results_path).name)} "
                f"CAUSAL_RUN_ID={_q(Path(causal_results_path).name)} "
                f"PRIMARY_RESULTS_DIR={_q(primary_results_path)} "
                f"CAUSAL_RESULTS_DIR={_q(causal_results_path)} "
                f"PRIMARY_GENERATED_DIR={_q(primary_generated_path)} "
                f"CAUSAL_GENERATED_DIR={_q(causal_generated_path)} "
                f"PRIMARY_AUDIT_SUMMARY_DIR={_q(primary_audit_path)} "
                f"CAUSAL_AUDIT_SUMMARY_DIR={_q(causal_audit_path)} "
                "AUDIT_SOURCE=open_judge "
                "bash scripts/aggregate_publication_human_audits.sh"
            ),
            detail=(
                "Complete the leakage-capable blinded annotator CSVs, or run the documented "
                "open local judge workflow before aggregation. Require result-source, "
                "export-protocol, judge-model, and prompt-template provenance to match the "
                "exact run artifacts."
            ),
        ),
        _step(
            "assess_claims",
            complete=bool(gates.get("claim_assessment_passed")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete"))
            and bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete")),
            command=(
                "uv run python scripts/assess_claims.py "
                f"--primary-results-dir {_q(primary_results_path)} "
                f"--causal-results-dir {_q(causal_results_path)} "
                f"--primary-audit-summary {_q(str(Path(primary_audit_path) / 'human_audit_summary.json'))} "
                f"--causal-audit-summary {_q(str(Path(causal_audit_path) / 'human_audit_summary.json'))} "
                f"--output-dir {_q(claim_generated_path)} "
                "--require-human-audit-support --require-cache-mediated-claim"
            ),
            detail="Gate the manuscript claim on H1, H2, H3, and declared audit support; do not rewrite thresholds after seeing results.",
        ),
        _step(
            "build_publication_bundle",
            complete=bool(status.get("publication_ready")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete"))
            and bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete"))
            and bool(gates.get("claim_assessment_passed")),
            command=(
                f"PRIMARY_RESULTS_DIR={_q(primary_results_path)} "
                f"CAUSAL_RESULTS_DIR={_q(causal_results_path)} "
                f"PRIMARY_GENERATED_DIR={_q(primary_generated_path)} "
                f"CAUSAL_GENERATED_DIR={_q(causal_generated_path)} "
                f"PRIMARY_AUDIT_SUMMARY_DIR={_q(primary_audit_path)} "
                f"CAUSAL_AUDIT_SUMMARY_DIR={_q(causal_audit_path)} "
                f"CLAIM_GENERATED_DIR={_q(claim_generated_path)} "
                f"ARXIV_SOURCE_DIR={_q(arxiv_source_path)} "
                f"ARXIV_ARCHIVE={_q(arxiv_archive_path)} "
                "bash scripts/build_publication_artifacts.sh"
            ),
            detail="Regenerate metrics, figures, tables, final PDF, and arXiv source bundle from recorded evidence.",
        ),
    ]
    next_step = next((step for step in steps if not step["complete"]), None)
    return {
        "schema_version": 1,
        "publication_ready": bool(status.get("publication_ready")),
        "blockers": status.get("blockers", []),
        "blocker_details": _blocker_details(status),
        "next_step": next_step["name"] if next_step else "done",
        "steps": steps,
    }


def _status_path(status: dict[str, Any], artifact_name: str, default: str) -> str:
    return _artifact_field(status, artifact_name, "path", default)


def _artifact_field(
    status: dict[str, Any],
    artifact_name: str,
    field_name: str,
    default: str,
) -> str:
    artifact = status.get(artifact_name)
    if not isinstance(artifact, dict):
        return default
    value = artifact.get(field_name)
    return str(value) if value else default


def _q(value: str) -> str:
    return shlex.quote(value)


def _raw_result_available(artifact: object) -> bool:
    if not isinstance(artifact, dict):
        return False
    missing = set(str(item) for item in artifact.get("missing", []))
    if missing - _DERIVED_RESULT_ARTIFACTS:
        return False
    disqualifiers = artifact.get("disqualifiers", [])
    if disqualifiers:
        return False
    readiness_failures = [str(item) for item in artifact.get("readiness_failures", [])]
    return all(
        _raw_readiness_failure_is_derived(failure) or _readiness_failure_is_ci_width(failure)
        for failure in readiness_failures
    )


def _prepared_result_available(artifact: object) -> bool:
    if not isinstance(artifact, dict):
        return False
    if artifact.get("missing") or artifact.get("disqualifiers"):
        return False
    readiness_failures = [str(item) for item in artifact.get("readiness_failures", [])]
    return all(_readiness_failure_is_ci_width(failure) for failure in readiness_failures)


def _raw_readiness_failure_is_derived(failure: str) -> bool:
    if failure in _DERIVED_READINESS_EXACT_FAILURES:
        return True
    return failure.startswith(_DERIVED_READINESS_PREFIXES)


def _has_ci_width_blockers(status: dict[str, Any]) -> bool:
    for artifact_name in ["primary_results", "causal_results"]:
        artifact = status.get(artifact_name)
        if not isinstance(artifact, dict):
            continue
        if any(
            _readiness_failure_is_ci_width(str(failure))
            for failure in artifact.get("readiness_failures", [])
        ):
            return True
    return False


def _readiness_failure_is_ci_width(failure: str) -> bool:
    normalized = failure.lower()
    return (
        "ci_width" in normalized
        or ("ci width" in normalized and "target" in normalized)
        or (" ci` width " in normalized and "target" in normalized)
        or ("_ci` width " in normalized and "target" in normalized)
    )


def _blocker_details(status: dict[str, Any]) -> dict[str, list[str]]:
    mapping = {
        "primary_results_complete": ("primary_results", ["missing", "disqualifiers", "readiness_failures"]),
        "causal_results_complete": ("causal_results", ["missing", "disqualifiers", "readiness_failures"]),
        "primary_human_audit_complete": ("primary_human_audit", ["missing", "failures"]),
        "causal_human_audit_complete": ("causal_human_audit", ["missing", "failures"]),
        "claim_assessment_passed": ("claim_assessment", ["failures"]),
        "paper_pdf_exists": ("paper_pdf", ["failure"]),
        "paper_pdf_valid": ("paper_pdf", ["failure"]),
        "arxiv_bundle_ready": ("arxiv_bundle", ["missing", "failures"]),
    }
    details: dict[str, list[str]] = {}
    for blocker in status.get("blockers", []):
        artifact_name, keys = mapping.get(str(blocker), ("", []))
        artifact = status.get(artifact_name) if artifact_name else None
        values: list[str] = []
        if isinstance(artifact, dict):
            for key in keys:
                raw_value = artifact.get(key)
                if isinstance(raw_value, list):
                    values.extend(str(item) for item in raw_value)
                elif raw_value:
                    values.append(str(raw_value))
        if values:
            details[str(blocker)] = sorted(set(values))
    return details


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-H200 Publication Next Steps",
        "",
        f"Publication ready: `{str(report['publication_ready']).lower()}`",
        f"Next step: `{report['next_step']}`",
        "",
        "## Blockers",
        "",
    ]
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
            for detail in report.get("blocker_details", {}).get(blocker, [])[:8]:
                lines.append(f"  - {detail}")
    else:
        lines.append("- none")
    lines.extend(["", "## Ordered Steps", ""])
    for step in report["steps"]:
        lines.extend(
            [
                f"### `{step['name']}`",
                f"- state: `{step['state']}`",
                f"- detail: {step['detail']}",
                f"- command: `{step['command']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _step(
    name: str,
    *,
    complete: bool,
    ready: bool,
    command: str,
    detail: str,
) -> dict[str, str]:
    if complete:
        state = "complete"
    elif ready:
        state = "ready"
    else:
        state = "blocked"
    return {
        "name": name,
        "state": state,
        "complete": complete,
        "command": command,
        "detail": detail,
    }


if __name__ == "__main__":
    main()
