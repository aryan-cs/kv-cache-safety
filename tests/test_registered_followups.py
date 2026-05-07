import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from plan_registered_followups import build_followup_plan, render_markdown


def test_followup_plan_requires_causal_extension_after_selective_only_result() -> None:
    plan = build_followup_plan(
        _assessment(h1=True, h2=True, h3=False, gate=False),
        causal_ci_power={"conservative_bernoulli_required_cluster_n": 401},
    )

    assert plan["status"] == "causal_gate_failed"
    assert plan["claim_gate_passed"] is False
    assert plan["registered_followups"][0]["name"] == "causal restoration extension"
    assert "401" in plan["registered_followups"][0]["success_criterion"]
    assert "not cache-mediated safety erasure" in plan["registered_followups"][0]["manuscript_scope"]
    assert any("Do not lower claim thresholds" in action for action in plan["prohibited_actions"])


def test_followup_plan_blocks_positive_claim_when_selectivity_fails() -> None:
    plan = build_followup_plan(
        _assessment(h1=True, h2=False, h3=False, gate=False),
        primary_ci_power={
            "conservative_bernoulli_required_cluster_n": 601,
            "conservative_ssei_two_component_required_cluster_n": 1201,
            "pilot_estimates": [{"estimated_required_cluster_n": 700}],
        },
    )

    assert plan["status"] == "selectivity_gate_failed"
    followup = plan["registered_followups"][0]
    assert followup["name"] == "powered selectivity extension"
    assert "`1201` prompt clusters" in followup["success_criterion"]
    assert "Do not describe a safety-erasure phenomenon" in followup["manuscript_scope"]


def test_followup_plan_requires_new_preregistration_when_h1_fails() -> None:
    plan = build_followup_plan(_assessment(h1=False, h2=False, h3=False, gate=False))

    assert plan["status"] == "no_positive_cache_safety_gate"
    assert "preregistered pivot" in plan["registered_followups"][0]["name"]
    assert "new hypothesis document" in plan["registered_followups"][0]["success_criterion"]


def test_followup_plan_marks_positive_claim_as_replication_only() -> None:
    plan = build_followup_plan(_assessment(h1=True, h2=True, h3=True, gate=True))

    assert plan["status"] == "positive_claim_supported"
    assert plan["claim_gate_passed"] is True
    assert plan["registered_followups"][0]["name"] == "model-family replication"


def test_followup_plan_markdown_is_plain_and_actionable() -> None:
    plan = build_followup_plan(_assessment(h1=True, h2=True, h3=False, gate=False))

    markdown = render_markdown(plan)

    assert "# Registered Follow-up Plan" in markdown
    assert "## Prohibited Actions" in markdown
    assert "Do not add unregistered suites" in markdown


def test_followup_plan_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(_assessment(h1=True, h2=True, h3=False, gate=False)))
    output_dir = tmp_path / "followups"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_registered_followups.py",
            "--claim-assessment",
            str(claim_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "registered_followup_plan.json").exists()
    assert (output_dir / "registered_followup_plan.md").exists()


def _assessment(*, h1: bool, h2: bool, h3: bool, gate: bool) -> dict:
    return {
        "claims": {
            "H1_behavioral_cache_sensitivity": {
                "passed": h1,
                "summary": "H1 summary",
                "best_evidence": {"key": "public_refusal_safety::kv_int4_sim"},
            },
            "H2_selective_safety_degradation": {
                "passed": h2,
                "summary": "H2 summary",
                "best_evidence": {"key": "kv_int4_sim"},
            },
            "H3_causal_safety_state_erasure": {
                "passed": h3,
                "summary": "H3 summary",
                "best_comparison": {"compressed_policy": "kv_int4_sim"},
            },
        },
        "human_audit_support": {"passed": gate},
        "publication_gate": {"passed": gate},
    }
