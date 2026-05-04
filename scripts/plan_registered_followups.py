from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan evidence-gated follow-up experiments without changing the registered claim "
            "scope or searching for publishable metrics post hoc."
        )
    )
    parser.add_argument("--claim-assessment", required=True, type=Path)
    parser.add_argument("--primary-ci-power", type=Path, default=None)
    parser.add_argument("--causal-ci-power", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    assessment = _load_json(args.claim_assessment)
    primary_ci = _load_optional_json(args.primary_ci_power)
    causal_ci = _load_optional_json(args.causal_ci_power)
    plan = build_followup_plan(
        assessment,
        primary_ci_power=primary_ci,
        causal_ci_power=causal_ci,
        source_paths={
            "claim_assessment": args.claim_assessment,
            "primary_ci_power": args.primary_ci_power,
            "causal_ci_power": args.causal_ci_power,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "registered_followup_plan.json", plan)
    (args.output_dir / "registered_followup_plan.md").write_text(
        render_markdown(plan), encoding="utf-8"
    )
    print(f"Wrote registered follow-up plan to {args.output_dir}")


def build_followup_plan(
    assessment: dict[str, Any],
    *,
    primary_ci_power: dict[str, Any] | None = None,
    causal_ci_power: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
) -> dict[str, Any]:
    claims = assessment.get("claims", {})
    h1 = claims.get("H1_behavioral_cache_sensitivity", {})
    h2 = claims.get("H2_selective_safety_degradation", {})
    h3 = claims.get("H3_causal_safety_state_erasure", {})
    audit = assessment.get("human_audit_support", {})
    publication_gate = assessment.get("publication_gate", {})

    if publication_gate.get("passed"):
        status = "positive_claim_supported"
        followups = _positive_claim_followups(audit)
    elif h1.get("passed") and h2.get("passed") and h3.get("passed"):
        status = "audit_blocked_positive_candidate"
        followups = _audit_blocked_followups()
    elif h1.get("passed") and h2.get("passed"):
        status = "causal_gate_failed"
        followups = _causal_gate_followups(h2, h3, causal_ci_power)
    elif h1.get("passed"):
        status = "selectivity_gate_failed"
        followups = _selectivity_gate_followups(h1, h2, primary_ci_power)
    else:
        status = "no_positive_cache_safety_gate"
        followups = _negative_or_pivot_followups(primary_ci_power)

    return {
        "schema_version": 1,
        "status": status,
        "claim_gate_passed": bool(publication_gate.get("passed")),
        "claim_state": {
            "H1_behavioral_cache_sensitivity": bool(h1.get("passed")),
            "H2_selective_safety_degradation": bool(h2.get("passed")),
            "H3_causal_safety_state_erasure": bool(h3.get("passed")),
            "human_audit_support": bool(audit.get("passed")),
        },
        "registered_followups": followups,
        "prohibited_actions": [
            "Do not lower claim thresholds after seeing results.",
            "Do not add unregistered suites or policies to the main claim without labeling them exploratory.",
            "Do not report smoke, mock, tiny-model, or incomplete-matrix runs as paper evidence.",
            "Do not claim cache-mediated safety erasure unless H1, H2, H3, and human-audit gates pass.",
        ],
        "source_artifacts": _source_artifacts(source_paths or {}),
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Registered Follow-up Plan",
        "",
        f"Status: `{plan['status']}`",
        f"Claim gate passed: `{str(plan['claim_gate_passed']).lower()}`",
        "",
        "## Claim State",
        "",
        "| Gate | Passed |",
        "| --- | --- |",
    ]
    for gate, passed in plan["claim_state"].items():
        lines.append(f"| {gate} | {str(passed).lower()} |")
    lines.extend(["", "## Registered Follow-ups", ""])
    for index, followup in enumerate(plan["registered_followups"], start=1):
        lines.extend(
            [
                f"{index}. **{followup['name']}**",
                f"   - Purpose: {followup['purpose']}",
                f"   - Trigger: {followup['trigger']}",
                f"   - Success criterion: {followup['success_criterion']}",
                f"   - Manuscript scope: {followup['manuscript_scope']}",
            ]
        )
    lines.extend(["", "## Prohibited Actions", ""])
    for action in plan["prohibited_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _positive_claim_followups(audit: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "name": "model-family replication",
            "purpose": "Test whether the supported phenomenon generalizes beyond the primary model family.",
            "trigger": "Run only after the registered claim gate passes on the primary and causal analyses.",
            "success_criterion": (
                "A non-primary open instruct model reproduces the direction of H1, H2, and H3 "
                "under the same thresholds or is reported as a bounded non-replication."
            ),
            "manuscript_scope": (
                "Without this replication, keep claims limited to the tested model family and "
                "report the human-audit support state as "
                f"`{str(bool(audit.get('passed'))).lower()}`."
            ),
        }
    ]


def _audit_blocked_followups() -> list[dict[str, str]]:
    return [
        {
            "name": "complete blinded human audit",
            "purpose": "Determine whether automated safety and leakage labels match human judgment.",
            "trigger": "Automated H1, H2, and H3 pass while the human-audit gate fails or is absent.",
            "success_criterion": (
                "Audit completion is 100%, unknown audit IDs are zero, and safety-direction "
                "human deltas do not contradict the automated effects."
            ),
            "manuscript_scope": (
                "Until this passes, describe the result as an automated-metric candidate rather "
                "than a publication-ready safety finding."
            ),
        }
    ]


def _causal_gate_followups(
    h2: dict[str, Any],
    h3: dict[str, Any],
    causal_ci_power: dict[str, Any] | None,
) -> list[dict[str, str]]:
    target = _best_key(h2)
    causal_gap = _best_summary(h3)
    required_n = _required_cluster_text(causal_ci_power)
    return [
        {
            "name": "causal restoration extension",
            "purpose": (
                "Test whether the strongest selective behavioral effect is actually carried by "
                "policy-relevant cache state rather than generic cache quality."
            ),
            "trigger": (
                f"H2 passed for `{target}`, but H3 did not clear the matched-control margin. "
                f"Current causal summary: {causal_gap}"
            ),
            "success_criterion": (
                "System-role restoration or policy-pinned retention clears the registered "
                "restoration-fraction threshold and exceeds matched user-role controls by the "
                f"registered margin. Plan prompt clusters using {required_n}."
            ),
            "manuscript_scope": (
                "Before this succeeds, frame the paper as selective cache-induced safety "
                "degradation, not cache-mediated safety erasure."
            ),
        }
    ]


def _selectivity_gate_followups(
    h1: dict[str, Any],
    h2: dict[str, Any],
    primary_ci_power: dict[str, Any] | None,
) -> list[dict[str, str]]:
    required_n = _required_cluster_text(primary_ci_power)
    return [
        {
            "name": "powered selectivity extension",
            "purpose": "Separate broad quality degradation from safety-specific degradation.",
            "trigger": (
                f"H1 passed with `{_best_key(h1)}`, but H2 failed. "
                f"Current selectivity summary: {_best_summary(h2)}"
            ),
            "success_criterion": (
                "The policy-level SSEI interval clears the preregistered threshold after adding "
                f"prompt clusters planned from {required_n}; otherwise report a broad cache "
                "robustness result or null selective effect."
            ),
            "manuscript_scope": (
                "Do not describe a safety-erasure phenomenon unless selectivity and causal gates "
                "also pass."
            ),
        }
    ]


def _negative_or_pivot_followups(primary_ci_power: dict[str, Any] | None) -> list[dict[str, str]]:
    required_n = _required_cluster_text(primary_ci_power)
    return [
        {
            "name": "negative-result confirmation or preregistered pivot",
            "purpose": (
                "Avoid post-hoc metric search while still allowing a scientifically meaningful "
                "follow-up if cache logs reveal a new mechanism."
            ),
            "trigger": (
                "H1 did not pass. Confirm the negative result with the planned prompt-cluster "
                f"count from {required_n}, or write a new preregistered mechanism before "
                "running any exploratory extension."
            ),
            "success_criterion": (
                "Either H1 remains negative and the manuscript reports a falsification result, "
                "or a new hypothesis document names the mechanism, suites, policies, thresholds, "
                "and exclusion criteria before additional runs start."
            ),
            "manuscript_scope": (
                "Any post-pivot evidence must be labeled as a separate registered analysis, not "
                "as if it were the original claim."
            ),
        }
    ]


def _best_key(claim: dict[str, Any]) -> str:
    evidence = claim.get("best_evidence") or claim.get("best_comparison") or {}
    return str(evidence.get("key") or evidence.get("compressed_policy") or "no eligible evidence")


def _best_summary(claim: dict[str, Any]) -> str:
    return str(claim.get("summary") or "No summary available.")


def _required_cluster_text(ci_power: dict[str, Any] | None) -> str:
    if not ci_power:
        return "the conservative CI plan"
    conservative = ci_power.get("conservative_bernoulli_required_cluster_n")
    estimates = ci_power.get("pilot_estimates") or []
    pilot_required = [
        int(row["estimated_required_cluster_n"])
        for row in estimates
        if row.get("estimated_required_cluster_n") is not None
    ]
    required = max([int(conservative or 0), *pilot_required], default=0)
    if required <= 0:
        return "the conservative CI plan"
    return f"`{required}` prompt clusters"


def _source_artifacts(source_paths: dict[str, Path | None]) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for label, path in source_paths.items():
        if path is None:
            continue
        artifacts[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return artifacts


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_json(path)


if __name__ == "__main__":
    main()
