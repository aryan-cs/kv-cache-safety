from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from report_publication_status import publication_status

from cache_safety_erasure.utils.io import write_json


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
        default=Path("paper/audit/h200_qwen_full_sweep_summary"),
    )
    parser.add_argument(
        "--causal-audit-dir",
        type=Path,
        default=Path("paper/audit/h200_causal_patch_qwen7b_summary"),
    )
    parser.add_argument(
        "--claim-assessment",
        type=Path,
        default=Path("paper/generated/claim_assessment/claim_assessment.json"),
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=Path("paper/cache_mediated_safety_erasure.pdf"),
    )
    parser.add_argument(
        "--arxiv-source-dir",
        type=Path,
        default=Path("paper/build/arxiv_source"),
    )
    parser.add_argument(
        "--arxiv-archive",
        type=Path,
        default=Path("paper/build/arxiv_source.tar.gz"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args()

    status = publication_status(
        primary_results_dir=args.primary_results_dir,
        causal_results_dir=args.causal_results_dir,
        primary_audit_dir=args.primary_audit_dir,
        causal_audit_dir=args.causal_audit_dir,
        claim_assessment_path=args.claim_assessment,
        paper_pdf=args.paper_pdf,
        arxiv_source_dir=args.arxiv_source_dir,
        arxiv_archive=args.arxiv_archive,
        require_arxiv_bundle=True,
    )
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
    steps = [
        _step(
            "complete_h200_results",
            complete=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete")),
            ready=True,
            command="bash scripts/wait_and_run_h200_sweep.sh",
            detail="Run or wait for the registered H200 launcher until primary and causal result directories pass artifact gates.",
        ),
        _step(
            "complete_human_audits",
            complete=bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete")),
            command="bash scripts/export_publication_audit_samples.sh && uv run python scripts/aggregate_human_audit.py --audit-csv paper/audit/<run_id>_audit_blinded_annotator_*.csv --key-jsonl paper/audit/<run_id>_audit_key.jsonl --results-dir results/<run_id> --export-manifest paper/audit/<run_id>_audit_export_manifest.json --output-dir paper/audit/<run_id>_summary",
            detail="Regenerate leakage-capable blinded templates, complete annotations, aggregate them, and require result-source and export-protocol hashes to match the exact run artifacts.",
        ),
        _step(
            "assess_claims",
            complete=bool(gates.get("claim_assessment_passed")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete"))
            and bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete")),
            command="uv run python scripts/assess_claims.py --primary-results-dir results/h200_qwen_full_sweep --causal-results-dir results/h200_causal_patch_qwen7b --primary-audit-summary paper/audit/h200_qwen_full_sweep_summary/human_audit_summary.json --causal-audit-summary paper/audit/h200_causal_patch_qwen7b_summary/human_audit_summary.json --output-dir paper/generated/claim_assessment --require-human-audit-support --require-cache-mediated-claim",
            detail="Gate the manuscript claim on H1, H2, H3, and human-audit support; do not rewrite thresholds after seeing results.",
        ),
        _step(
            "build_publication_bundle",
            complete=bool(status.get("publication_ready")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete"))
            and bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete"))
            and bool(gates.get("claim_assessment_passed")),
            command="bash scripts/build_publication_artifacts.sh",
            detail="Regenerate metrics, figures, tables, final PDF, and arXiv source bundle from recorded evidence.",
        ),
    ]
    next_step = next((step for step in steps if not step["complete"]), None)
    return {
        "schema_version": 1,
        "publication_ready": bool(status.get("publication_ready")),
        "blockers": status.get("blockers", []),
        "next_step": next_step["name"] if next_step else "done",
        "steps": steps,
    }


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
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
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
