from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from aggregate_human_audit import (
    render_deltas_latex,
    render_summary_latex,
    render_summary_markdown,
)
from assess_claims import (
    assess_claims,
    render_abstract_status_latex,
    render_interpretation_latex,
    render_latex_table,
)
from check_final_pdf_text import (
    extract_pdf_text,
    final_pdf_text_failures,
)
from check_human_audit_readiness import (
    DEFAULT_REQUIRED_LABELS,
    check_audit_input_source_match,
    check_audit_summary_source_match,
    check_human_audit_readiness,
)
from check_publication_readiness import _check_figure_manifest, _pdf_page_failure
from package_arxiv_submission import (
    FIGURE_SOURCES,
    _final_source_failures,
    _invalid_arxiv_support_files,
    _rewrite_failures,
)

from cache_safety_erasure.utils.io import file_sha256, write_json

REQUIRED_RUN_ARTIFACTS = [
    "config.resolved.yaml",
    "environment.json",
    "manifest.json",
    "prompts.jsonl",
    "generations.jsonl",
    "metrics.json",
    "cache_stats.parquet",
    "figures/manifest.json",
]
REQUIRED_AUDIT_ARTIFACTS = [
    "audit_manifest.json",
    "human_audit_summary.json",
    "human_audit_summary.md",
    "human_audit_summary_table.tex",
    "human_audit_deltas_table.tex",
]
ACTIVE_ARXIV_BUNDLE_FILES = [
    "generated/active_primary/main_results_table.tex",
    "generated/active_primary/suite_level_effects_table.tex",
    "generated/active_primary/result_macros.tex",
    "generated/active_causal/causal_restoration_table.tex",
    "generated/active_causal/result_macros.tex",
]
ACTIVE_ARXIV_AUDIT_FILES = [
    "audit/active_primary_summary/human_audit_summary_table.tex",
    "audit/active_primary_summary/human_audit_deltas_table.tex",
    "audit/active_causal_summary/human_audit_summary_table.tex",
    "audit/active_causal_summary/human_audit_deltas_table.tex",
]
STATIC_ARXIV_BUNDLE_FILES = [
    "generated/claim_assessment/abstract_status_sentence.tex",
    "generated/claim_assessment/claim_assessment_table.tex",
    "generated/claim_assessment/claim_interpretation.tex",
]
REQUIRED_ARXIV_BUNDLE_FILES = [
    "generated/h200_qwen_full_sweep/main_results_table.tex",
    "generated/h200_qwen_full_sweep/suite_level_effects_table.tex",
    "generated/h200_qwen_full_sweep/result_macros.tex",
    "generated/h200_causal_patch_qwen7b/causal_restoration_table.tex",
    "generated/h200_causal_patch_qwen7b/result_macros.tex",
    *ACTIVE_ARXIV_BUNDLE_FILES,
    *ACTIVE_ARXIV_AUDIT_FILES,
    *STATIC_ARXIV_BUNDLE_FILES,
]
REQUIRED_ARXIV_FIGURE_FILES = [f"figures/{name}" for name in FIGURE_SOURCES]
PRIMARY_REQUIRED_FIGURES = [
    "safety_capability_phase_portrait",
    "selective_safety_erasure_heatmap",
    "prompt_effect_constellation",
    "cache_state_fingerprint",
    "safety_state_atlas",
    "policy_uncertainty_braid",
]
CAUSAL_REQUIRED_FIGURES = [
    "causal_restoration_fraction",
    "causal_restoration_flow",
]
PRIMARY_REQUIRED_SUITES = [
    "system_leakage",
    "public_system_leakage",
    "public_refusal_safety",
    "public_benign_overrefusal",
    "public_xstest_safe",
    "public_capability_arc",
]


def _required_arxiv_bundle_files(
    primary_generated_dir: Path = Path("paper/generated/h200_qwen_full_sweep"),
    causal_generated_dir: Path = Path("paper/generated/h200_causal_patch_qwen7b"),
    primary_audit_dir: Path = Path("paper/audit/h200_qwen_full_sweep_summary"),
    causal_audit_dir: Path = Path("paper/audit/h200_causal_patch_qwen7b_summary"),
) -> list[str]:
    primary_name = primary_generated_dir.name
    causal_name = causal_generated_dir.name
    primary_audit_name = primary_audit_dir.name
    causal_audit_name = causal_audit_dir.name
    return [
        f"generated/{primary_name}/main_results_table.tex",
        f"generated/{primary_name}/suite_level_effects_table.tex",
        f"generated/{primary_name}/result_macros.tex",
        f"generated/{causal_name}/causal_restoration_table.tex",
        f"generated/{causal_name}/result_macros.tex",
        *ACTIVE_ARXIV_BUNDLE_FILES,
        f"audit/{primary_audit_name}/human_audit_summary_table.tex",
        f"audit/{primary_audit_name}/human_audit_deltas_table.tex",
        f"audit/{causal_audit_name}/human_audit_summary_table.tex",
        f"audit/{causal_audit_name}/human_audit_deltas_table.tex",
        *ACTIVE_ARXIV_AUDIT_FILES,
        *STATIC_ARXIV_BUNDLE_FILES,
    ]


CAUSAL_REQUIRED_SUITES = [
    "system_leakage",
    "public_system_leakage",
    "public_refusal_safety",
]
PRIMARY_SUITE_MIN_PROMPTS = {
    "system_leakage": 2,
    "public_xstest_safe": 200,
}
CAUSAL_SUITE_MIN_PROMPTS = {
    "system_leakage": 2,
}
PRIMARY_REQUIRED_POLICIES = [
    "none",
    "sliding_window",
    "sink_recent",
    "random_matched",
    "kv_int8_sim",
    "kv_int4_sim",
    "policy_pinned",
]
CAUSAL_REQUIRED_POLICIES = ["none", "kv_int4_sim", "policy_pinned"]
PRIMARY_MAX_CI_WIDTH = 0.08
CAUSAL_MAX_CI_WIDTH = 0.12
PROFILE_CONTRACTS = {
    "primary": {
        "run_name": "h200_qwen_full_sweep",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
    },
    "causal": {
        "run_name": "h200_causal_patch_qwen7b",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
    },
}
MIN_H200_DEVICE_MEMORY_BYTES = 100 * 1024**3
EXPECTED_CLAIM_IDS = [
    "H1_behavioral_cache_sensitivity",
    "H2_selective_safety_degradation",
    "H3_causal_safety_state_erasure",
]
EXPECTED_PUBLICATION_REQUIRED_CLAIMS = [*EXPECTED_CLAIM_IDS, "human_audit_support"]
RAW_EVIDENCE_BASENAMES = {
    "audit_key.jsonl",
    "audit_labels.csv",
    "audit_sample.jsonl",
    "cache_stats.parquet",
    "generations.jsonl",
    "prompts.jsonl",
}
RAW_EVIDENCE_SUFFIXES = {".csv", ".jsonl", ".parquet"}
FINAL_PDF_NAME = "cache_mediated_safety_erasure.pdf"
FINAL_PDF_REQUIRED_SOURCE_PREFIXES = [
    "latex_main",
    "bibliography",
    "primary_results",
    "causal_results",
    "primary_generated",
    "causal_generated",
    "claim_",
    "primary_audit",
    "causal_audit",
    "primary_figure",
    "causal_figure",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report publication-blocking artifact and claim-gate status."
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
        "--primary-generated-dir",
        type=Path,
        default=Path("paper/generated/h200_qwen_full_sweep"),
    )
    parser.add_argument(
        "--causal-generated-dir",
        type=Path,
        default=Path("paper/generated/h200_causal_patch_qwen7b"),
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=Path("paper/cache_mediated_safety_erasure.pdf"),
    )
    parser.add_argument(
        "--allow-missing-paper-pdf",
        action="store_true",
        help="Permit a missing PDF when checking readiness before rebuilding the final PDF.",
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
    parser.add_argument(
        "--require-arxiv-bundle",
        action="store_true",
        help="Require the final arXiv source directory and archive to be complete.",
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
        primary_generated_dir=args.primary_generated_dir,
        causal_generated_dir=args.causal_generated_dir,
        paper_pdf=args.paper_pdf,
        require_paper_pdf=not args.allow_missing_paper_pdf,
        arxiv_source_dir=args.arxiv_source_dir,
        arxiv_archive=args.arxiv_archive,
        require_arxiv_bundle=args.require_arxiv_bundle,
    )
    if args.output_json is not None:
        write_json(args.output_json, status)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(status), encoding="utf-8")
    print(render_markdown(status))
    if args.fail_if_not_ready and not status["publication_ready"]:
        raise SystemExit(1)


def publication_status(
    *,
    primary_results_dir: Path,
    causal_results_dir: Path,
    primary_audit_dir: Path,
    causal_audit_dir: Path,
    claim_assessment_path: Path,
    paper_pdf: Path,
    primary_generated_dir: Path = Path("paper/generated/h200_qwen_full_sweep"),
    causal_generated_dir: Path = Path("paper/generated/h200_causal_patch_qwen7b"),
    require_paper_pdf: bool = True,
    arxiv_source_dir: Path = Path("paper/build/arxiv_source"),
    arxiv_archive: Path = Path("paper/build/arxiv_source.tar.gz"),
    require_arxiv_bundle: bool = False,
) -> dict[str, Any]:
    primary = _run_status(primary_results_dir, profile="primary")
    causal = _run_status(causal_results_dir, profile="causal")
    primary_audit = _audit_status(primary_audit_dir, primary_results_dir)
    causal_audit = _audit_status(causal_audit_dir, causal_results_dir)
    claim_assessment = _claim_status(
        claim_assessment_path,
        primary_results_dir=primary_results_dir,
        causal_results_dir=causal_results_dir,
        primary_audit_dir=primary_audit_dir,
        causal_audit_dir=causal_audit_dir,
    )
    pdf = _pdf_status(
        paper_pdf,
        expected_sources=_final_pdf_expected_sources(
            primary_results_dir=primary_results_dir,
            causal_results_dir=causal_results_dir,
            primary_audit_dir=primary_audit_dir,
            causal_audit_dir=causal_audit_dir,
            primary_generated_dir=primary_generated_dir,
            causal_generated_dir=causal_generated_dir,
            claim_assessment_path=claim_assessment_path,
        ),
    )
    arxiv = _arxiv_status(
        arxiv_source_dir,
        arxiv_archive,
        primary_generated_dir=primary_generated_dir,
        causal_generated_dir=causal_generated_dir,
        primary_audit_dir=primary_audit_dir,
        causal_audit_dir=causal_audit_dir,
    )

    gates = {
        "primary_results_complete": primary["complete"],
        "causal_results_complete": causal["complete"],
        "primary_human_audit_complete": primary_audit["complete"],
        "causal_human_audit_complete": causal_audit["complete"],
        "claim_assessment_passed": claim_assessment["passed"],
        "paper_pdf_exists": pdf["exists"] or not require_paper_pdf,
        "paper_pdf_valid": pdf["valid"] or not require_paper_pdf,
    }
    if require_arxiv_bundle:
        gates["arxiv_bundle_ready"] = arxiv["complete"]
    blockers = [gate for gate, passed in gates.items() if not passed]
    release_gates = {**gates, "arxiv_bundle_ready": arxiv["complete"]}
    release_blockers = [gate for gate, passed in release_gates.items() if not passed]
    evidence_gate_names = [
        "primary_results_complete",
        "causal_results_complete",
        "primary_human_audit_complete",
        "causal_human_audit_complete",
        "claim_assessment_passed",
    ]
    evidence_blockers = [gate for gate in evidence_gate_names if not gates[gate]]
    return {
        "schema_version": 1,
        "publication_ready": not blockers,
        "blockers": blockers,
        "release_ready": not release_blockers,
        "release_blockers": release_blockers,
        "evidence_ready": not evidence_blockers,
        "evidence_blockers": evidence_blockers,
        "gates": gates,
        "primary_results": primary,
        "causal_results": causal,
        "primary_human_audit": primary_audit,
        "causal_human_audit": causal_audit,
        "claim_assessment": claim_assessment,
        "paper_pdf": pdf,
        "paper_pdf_required": require_paper_pdf,
        "arxiv_bundle": arxiv,
        "arxiv_bundle_required": require_arxiv_bundle,
    }


def render_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# Publication Status",
        "",
        f"Publication ready: `{str(status['publication_ready']).lower()}`",
        f"Release ready: `{str(status.get('release_ready', False)).lower()}`",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for gate, passed in status["gates"].items():
        lines.append(f"| `{gate}` | {'pass' if passed else 'fail'} |")
    lines.extend(["", "## Blockers", ""])
    if status["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in status["blockers"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            _artifact_line("primary results", status["primary_results"]),
            _artifact_line("causal results", status["causal_results"]),
            _artifact_line(
                _audit_artifact_label("primary", status["primary_human_audit"]),
                status["primary_human_audit"],
            ),
            _artifact_line(
                _audit_artifact_label("causal", status["causal_human_audit"]),
                status["causal_human_audit"],
            ),
            _claim_line(status["claim_assessment"]),
            _pdf_line(status["paper_pdf"], evidence_ready=status["evidence_ready"]),
            _arxiv_line(status["arxiv_bundle"]),
            "",
        ]
    )
    return "\n".join(lines)


def _run_status(results_dir: Path, *, profile: str) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (results_dir / name).exists()]
    manifest = _read_json(results_dir / "manifest.json")
    environment = _read_json(results_dir / "environment.json")
    metrics = _read_json(results_dir / "metrics.json")
    disqualifiers: list[str] = []
    if manifest:
        model_provider = str(manifest.get("model_provider", ""))
        model_id = str(manifest.get("model_id", ""))
        run_name = str(manifest.get("run_name", ""))
        if manifest.get("git_dirty"):
            disqualifiers.append("dirty_git_tree")
        if model_provider == "mock":
            disqualifiers.append("mock_model")
        if "tiny" in model_id.lower():
            disqualifiers.append("tiny_model")
        if "smoke" in run_name.lower() or "smoke" in results_dir.name.lower():
            disqualifiers.append("smoke_run")
    readiness_failures = _run_readiness_failures(results_dir, manifest, profile=profile)
    if manifest:
        readiness_failures.extend(
            _profile_identity_failures(
                manifest,
                environment,
                profile=profile,
                results_dir=results_dir,
            )
        )
    return {
        "path": str(results_dir),
        "complete": not missing and not disqualifiers and not readiness_failures,
        "missing": missing,
        "disqualifiers": disqualifiers,
        "readiness_failures": readiness_failures,
        "manifest_present": bool(manifest),
        "metrics_present": bool(metrics),
        "model_id": manifest.get("model_id") if manifest else None,
        "git_commit": manifest.get("git_commit") if manifest else None,
        "expected_generation_count": manifest.get("expected_generation_count") if manifest else None,
        "policy_count": len(manifest.get("cache_policy_labels", [])) if manifest else None,
        "prompt_counts": manifest.get("prompt_counts") if manifest else None,
    }


def _run_readiness_failures(
    results_dir: Path, manifest: dict[str, Any], *, profile: str
) -> list[str]:
    if not manifest:
        return []
    failures = []
    if not manifest.get("cache_policy_configs"):
        failures.append("manifest_lacks_cache_policy_configs")
    if not manifest.get("cache_policy_labels"):
        failures.append("manifest_lacks_cache_policy_labels")
    if manifest.get("expected_generation_count") is None:
        failures.append("manifest_lacks_expected_generation_count")
    else:
        generation_count = _jsonl_row_count(results_dir / "generations.jsonl")
        if generation_count is not None and generation_count != int(
            manifest["expected_generation_count"]
        ):
            failures.append(
                f"generation_row_count={generation_count}; expected={manifest['expected_generation_count']}"
            )
    if not manifest.get("prompt_counts"):
        failures.append("manifest_lacks_prompt_counts")
    failures.extend(_profile_contract_failures(results_dir, manifest, profile=profile))
    failures.extend(_prompt_generation_matrix_failures(results_dir, manifest))
    failures.extend(_prompt_suite_provenance_failures(results_dir, manifest))
    failures.extend(_figure_source_failures(results_dir))
    failures.extend(_figure_manifest_failures(results_dir, profile=profile))
    return failures


def _profile_identity_failures(
    manifest: dict[str, Any],
    environment: dict[str, Any],
    *,
    profile: str,
    results_dir: Path,
) -> list[str]:
    contract = PROFILE_CONTRACTS.get(profile, {})
    failures = []
    for key in ["run_name", "model_id"]:
        expected = contract.get(key)
        observed = manifest.get(key)
        if not expected:
            continue
        if key == "run_name" and profile == "primary":
            expected_run_names = _expected_primary_run_names(manifest, results_dir, str(expected))
            if observed not in expected_run_names:
                expected_text = ", ".join(sorted(repr(name) for name in expected_run_names))
                failures.append(f"{key}={observed!r}; expected one of {expected_text}")
            continue
        if observed != expected:
            failures.append(f"{key}={observed!r}; expected={expected!r}")
    if not manifest.get("config_sha256"):
        failures.append("manifest_lacks_config_sha256")
    if environment.get("git_commit") and manifest.get("git_commit") != environment.get("git_commit"):
        failures.append("manifest_environment_git_commit_mismatch")
    failures.extend(_run_commit_history_failures(manifest))
    failures.extend(_combined_source_commit_failures(manifest))
    if environment.get("git_dirty"):
        failures.append("dirty_environment_git_tree")
    if environment:
        if environment.get("torch_cuda_available") is not True:
            failures.append("environment_lacks_cuda")
        cuda_devices = environment.get("cuda_devices") or []
        h200_devices = [
            device
            for device in cuda_devices
            if "h200" in str(device.get("name", "")).lower()
            and int(device.get("total_memory") or 0) >= MIN_H200_DEVICE_MEMORY_BYTES
        ]
        if not h200_devices:
            failures.append("environment_lacks_h200_gpu")
    return failures


def _expected_primary_run_names(
    manifest: dict[str, Any], results_dir: Path, default_run_name: str
) -> set[str]:
    expected = {default_run_name}
    combined_results = manifest.get("combined_results")
    if not isinstance(combined_results, dict):
        return expected
    merged_run_name = str(combined_results.get("merged_run_name") or "")
    if merged_run_name and merged_run_name == results_dir.name:
        expected.add(merged_run_name)
    output_results_dir = str(combined_results.get("output_results_dir") or "")
    output_run_name = Path(output_results_dir).name if output_results_dir else ""
    if output_run_name and output_run_name == results_dir.name:
        expected.add(output_run_name)
    return expected


def _jsonl_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _profile_contract_failures(
    results_dir: Path, manifest: dict[str, Any], *, profile: str
) -> list[str]:
    failures = []
    required_suites = PRIMARY_REQUIRED_SUITES if profile == "primary" else CAUSAL_REQUIRED_SUITES
    suite_min_prompts = (
        PRIMARY_SUITE_MIN_PROMPTS if profile == "primary" else CAUSAL_SUITE_MIN_PROMPTS
    )
    default_min_prompts = 600
    prompt_counts = manifest.get("prompt_counts") or {}
    for suite in required_suites:
        if suite not in prompt_counts:
            failures.append(f"missing_required_suite:{suite}")
            continue
        required_count = suite_min_prompts.get(suite, default_min_prompts)
        observed_count = _coerce_nonnegative_int(prompt_counts.get(suite), default=0)
        if observed_count < required_count:
            failures.append(f"suite_prompt_count:{suite}={observed_count}; required={required_count}")

    policy_configs = manifest.get("cache_policy_configs") or []
    policy_names = {
        str(policy.get("name"))
        for policy in policy_configs
        if isinstance(policy, dict) and policy.get("name")
    }
    required_policies = (
        PRIMARY_REQUIRED_POLICIES if profile == "primary" else CAUSAL_REQUIRED_POLICIES
    )
    for policy in required_policies:
        if not _policy_present(policy, policy_configs, policy_names):
            failures.append(f"missing_required_policy:{policy}")
    metrics = _read_json(results_dir / "metrics.json")
    if profile == "causal":
        failures.extend(_causal_patch_contract_failures(policy_configs))
        if metrics and not metrics.get("causal_restoration"):
            failures.append("missing_causal_restoration_metrics")
    failures.extend(_ci_width_failures(metrics, profile=profile))
    return failures


def _policy_present(
    required_policy: str, policy_configs: list[Any], policy_names: set[str]
) -> bool:
    return required_policy in policy_names or any(
        str(policy.get("name", "")).startswith(required_policy)
        for policy in policy_configs
        if isinstance(policy, dict)
    )


def _causal_patch_contract_failures(policy_configs: list[Any]) -> list[str]:
    patched = [
        policy.get("patch_from_baseline")
        for policy in policy_configs
        if isinstance(policy, dict) and isinstance(policy.get("patch_from_baseline"), dict)
    ]
    if not patched:
        return ["missing_causal_patch_policy"]
    has_system = any("system" in (patch.get("token_roles") or []) for patch in patched)
    has_user_control = any(
        "user" in (patch.get("token_roles") or [])
        and "system" in (patch.get("match_token_count_to_roles") or [])
        for patch in patched
    )
    system_signatures = {
        _patch_control_signature(patch)
        for patch in patched
        if "system" in (patch.get("token_roles") or [])
    }
    user_control_signatures = {
        _patch_control_signature(patch)
        for patch in patched
        if "user" in (patch.get("token_roles") or [])
        and "system" in (patch.get("match_token_count_to_roles") or [])
    }
    failures = []
    if not has_system:
        failures.append("missing_system_patch_policy")
    if not has_user_control:
        failures.append("missing_matched_user_control_patch_policy")
    if system_signatures and user_control_signatures and not (
        system_signatures & user_control_signatures
    ):
        failures.append("missing_same_signature_matched_user_control_patch_policy")
    return failures


def _patch_control_signature(patch: dict[str, Any]) -> tuple[object, ...]:
    return (
        tuple(_as_string_list(patch.get("components") or ["key", "value"])),
        "" if patch.get("max_tokens") is None else str(patch.get("max_tokens")),
        str(patch.get("selection") or ""),
        tuple(_as_string_list(patch.get("token_indices"))),
        tuple(_as_string_list(patch.get("layers"))),
        tuple(_as_string_list(patch.get("heads"))),
    )


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _run_commit_history_failures(manifest: dict[str, Any]) -> list[str]:
    commit = str(manifest.get("git_commit") or "").strip()
    if not _looks_like_git_commit(commit):
        return []
    return _commit_history_failures(
        commit,
        missing_message=f"run_git_commit_not_in_analysis_checkout:{commit}",
        ancestor_message=f"run_git_commit_not_ancestor_of_analysis_head:{commit}",
    )


def _combined_source_commit_failures(manifest: dict[str, Any]) -> list[str]:
    combined_results = manifest.get("combined_results")
    if not isinstance(combined_results, dict):
        return []
    failures: list[str] = []
    for label in ["base", "extension"]:
        if f"{label}_git_dirty" not in combined_results:
            failures.append(f"combined_source_lacks_git_dirty:{label}")
        if combined_results.get(f"{label}_git_dirty"):
            failures.append(f"combined_source_dirty_git_tree:{label}")
        commit = str(combined_results.get(f"{label}_git_commit") or "").strip()
        if not commit:
            failures.append(f"combined_source_lacks_git_commit:{label}")
            continue
        if not _looks_like_git_commit(commit):
            continue
        failures.extend(
            _commit_history_failures(
                commit,
                missing_message=f"combined_source_git_commit_not_in_analysis_checkout:{label}:{commit}",
                ancestor_message=f"combined_source_git_commit_not_ancestor_of_analysis_head:{label}:{commit}",
            )
        )
    return failures


def _commit_history_failures(
    commit: str, *, missing_message: str, ancestor_message: str
) -> list[str]:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if exists.returncode != 0:
        return [missing_message]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        return [ancestor_message]
    return []


def _looks_like_git_commit(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", value))


def _ci_width_failures(metrics: dict[str, Any], *, profile: str) -> list[str]:
    if not metrics:
        return []
    max_width = PRIMARY_MAX_CI_WIDTH if profile == "primary" else CAUSAL_MAX_CI_WIDTH
    failures = []
    for policy, contrast in (metrics.get("policy_level_contrasts") or {}).items():
        if str(policy) == "none" or not isinstance(contrast, dict):
            continue
        ci = contrast.get("selective_safety_erasure_index_ci") or {}
        failures.extend(
            _ci_interval_failures(
                f"{policy}:policy_level_ssei_ci",
                ci,
                max_width=max_width,
            )
        )
    for key, value in metrics.get("selective_safety_erasure", {}).items():
        if not isinstance(value, dict):
            failures.append(f"{key}:malformed_selective_safety_erasure_entry")
            continue
        ci = value.get("paired_safety_degradation_ci") or {}
        failures.extend(_ci_interval_failures(f"{key}:paired_safety_ci", ci, max_width=max_width))
    if profile == "causal":
        for key, value in (metrics.get("causal_restoration") or {}).items():
            if not isinstance(value, dict):
                failures.append(f"{key}:malformed_causal_restoration_entry")
                continue
            for metric in [
                "safety_restoration_fraction_ci",
                "refusal_restoration_fraction_ci",
                "leakage_avoidance_restoration_fraction_ci",
            ]:
                point_metric = metric.removesuffix("_ci")
                if point_metric in value or metric in value:
                    ci = value.get(metric) or {}
                    failures.extend(
                        _ci_interval_failures(f"{key}:{metric}", ci, max_width=max_width)
                    )
    return failures


def _ci_interval_failures(label: str, ci: dict[str, Any], *, max_width: float) -> list[str]:
    if not isinstance(ci, dict):
        return [f"{label}:invalid_ci"]
    ci_low = ci.get("ci_low")
    ci_high = ci.get("ci_high")
    if ci_low is None or ci_high is None:
        return [f"{label}:missing_ci"]
    try:
        low = float(ci_low)
        high = float(ci_high)
    except (TypeError, ValueError):
        return [f"{label}:invalid_ci"]
    if not math.isfinite(low) or not math.isfinite(high):
        return [f"{label}:nonfinite_ci"]
    width = high - low
    if width < 0:
        return [f"{label}:reversed_ci"]
    if width > max_width:
        return [f"{label}_width={width:.3f}; target<={max_width:.3f}"]
    return []


def _coerce_nonnegative_int(value: object, *, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _prompt_generation_matrix_failures(results_dir: Path, manifest: dict[str, Any]) -> list[str]:
    prompts_path = results_dir / "prompts.jsonl"
    generations_path = results_dir / "generations.jsonl"
    if not prompts_path.exists() or not generations_path.exists():
        return []
    prompt_rows, prompt_failures = _read_jsonl_objects(prompts_path, label="prompts")
    generation_rows, generation_failures = _read_jsonl_objects(
        generations_path, label="generations"
    )
    failures = [*prompt_failures, *generation_failures]
    if prompt_failures or generation_failures:
        return failures

    prompt_counts: dict[str, int] = {}
    prompt_keys: list[tuple[str, str]] = []
    malformed_prompts = 0
    for row in prompt_rows:
        suite = row.get("suite")
        prompt_id = row.get("prompt_id")
        if suite is None or prompt_id is None:
            malformed_prompts += 1
            continue
        suite_key = str(suite)
        prompt_key = str(prompt_id)
        prompt_counts[suite_key] = prompt_counts.get(suite_key, 0) + 1
        prompt_keys.append((suite_key, prompt_key))
    if malformed_prompts:
        failures.append(f"prompts_jsonl_malformed_matrix_keys:{malformed_prompts}")

    manifest_counts = {
        str(suite): _coerce_nonnegative_int(count, default=-1)
        for suite, count in (manifest.get("prompt_counts") or {}).items()
    }
    for suite in sorted(set(manifest_counts) | set(prompt_counts)):
        if prompt_counts.get(suite, 0) != manifest_counts.get(suite, -1):
            failures.append(
                f"prompt_count_mismatch:{suite}={prompt_counts.get(suite, 0)}; "
                f"manifest={manifest_counts.get(suite)}"
            )

    policy_labels = [str(label) for label in manifest.get("cache_policy_labels") or []]
    seeds = [_coerce_nonnegative_int(seed, default=-1) for seed in manifest.get("seeds") or []]
    if not policy_labels or not seeds or malformed_prompts:
        return failures

    expected_keys = {
        (suite, prompt_id, policy, seed)
        for suite, prompt_id in prompt_keys
        for policy in policy_labels
        for seed in seeds
    }
    observed_keys = []
    malformed_generations = 0
    for row in generation_rows:
        try:
            observed_keys.append(
                (
                    str(row["suite"]),
                    str(row["prompt_id"]),
                    str(row["policy"]),
                    int(row["seed"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            malformed_generations += 1
    if malformed_generations:
        failures.append(f"generations_jsonl_malformed_matrix_keys:{malformed_generations}")
    observed_key_set = set(observed_keys)
    duplicate_count = len(observed_keys) - len(observed_key_set)
    if duplicate_count:
        failures.append(f"generation_matrix_duplicate_rows:{duplicate_count}")
    missing = expected_keys - observed_key_set
    extra = observed_key_set - expected_keys
    if missing:
        failures.append(
            f"generation_matrix_missing_rows:{len(missing)}; "
            f"first={_format_matrix_key(sorted(missing)[0])}"
        )
    if extra:
        failures.append(
            f"generation_matrix_extra_rows:{len(extra)}; "
            f"first={_format_matrix_key(sorted(extra)[0])}"
        )
    return failures


def _read_jsonl_objects(path: Path, *, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    failures = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"invalid_{label}_jsonl:{line_number}:{exc.msg}")
                continue
            if not isinstance(row, dict):
                failures.append(f"invalid_{label}_jsonl:{line_number}:non_object")
                continue
            rows.append(row)
    return rows, failures


def _format_matrix_key(key: tuple[str, str, str, int]) -> str:
    suite, prompt_id, policy, seed = key
    return f"suite={suite},prompt_id={prompt_id},policy={policy},seed={seed}"


def _prompt_suite_provenance_failures(results_dir: Path, manifest: dict[str, Any]) -> list[str]:
    prompt_counts = manifest.get("prompt_counts") or {}
    prompt_suites = manifest.get("prompt_suites") or list(prompt_counts)
    public_suites = sorted(str(suite) for suite in prompt_suites if str(suite).startswith("public_"))
    if not public_suites:
        return []
    failures = []
    prompt_suite_manifests = manifest.get("prompt_suite_manifests") or {}
    for suite in public_suites:
        suite_manifest = prompt_suite_manifests.get(suite)
        if not isinstance(suite_manifest, dict):
            failures.append(f"missing_processed_suite_manifest:{suite}")
            continue
        if not suite_manifest.get("sha256") or not suite_manifest.get("record_count"):
            failures.append(f"processed_suite_manifest_lacks_hash_count:{suite}")

    prompts_path = results_dir / "prompts.jsonl"
    if not prompts_path.exists():
        return failures
    public_without_provenance = 0
    public_without_precise_provenance = 0
    with prompts_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"invalid_prompts_jsonl:{line_number}:{exc.msg}")
                continue
            if str(row.get("suite", "")).startswith("public_"):
                metadata = row.get("metadata") or {}
                if not metadata.get("source_dataset") or not metadata.get("source_split"):
                    public_without_provenance += 1
                if _public_prompt_precise_provenance_failure(row, metadata):
                    public_without_precise_provenance += 1
    if public_without_provenance:
        failures.append(f"public_prompts_lack_dataset_provenance:{public_without_provenance}")
    if public_without_precise_provenance:
        failures.append(
            "public_prompts_lack_precise_dataset_provenance:"
            f"{public_without_precise_provenance}"
        )
    return failures


def _public_prompt_precise_provenance_failure(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    required_value_fields = [
        "source_dataset",
        "source_split",
        "source_revision",
        "source_fingerprint",
        "source_version",
    ]
    if any(not metadata.get(field) for field in required_value_fields):
        return True
    required_present_fields = [
        "source_config",
        "source_config_name",
        "source_homepage",
        "source_license",
    ]
    if any(field not in metadata for field in required_present_fields):
        return True
    source_locator_fields = [
        "source_id",
        "source_row_index",
        "source_group_id",
        "xstest_task_id",
    ]
    if any(metadata.get(field) not in {None, ""} for field in source_locator_fields):
        return False
    return not re.search(r"_\d{6}$", str(row.get("prompt_id") or ""))


def _figure_source_failures(results_dir: Path) -> list[str]:
    manifest = _read_json(results_dir / "figures" / "manifest.json")
    if not manifest:
        return []
    source_artifacts = manifest.get("source_artifacts") or {}
    failures = []
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        source = source_artifacts.get(name)
        if not isinstance(source, dict):
            failures.append(f"figures_manifest_lacks_source:{name}")
            continue
        source_path = results_dir / name
        if not source_path.exists():
            failures.append(f"figures_manifest_source_missing:{name}")
            continue
        if source.get("sha256") != file_sha256(source_path):
            failures.append(f"figures_manifest_source_hash_stale:{name}")
    return failures


def _figure_manifest_failures(results_dir: Path, *, profile: str) -> list[str]:
    failures: list[str] = []
    required_figures = PRIMARY_REQUIRED_FIGURES if profile == "primary" else CAUSAL_REQUIRED_FIGURES
    _check_figure_manifest(
        results_dir / "figures",
        results_dir,
        failures,
        require_causal_patch=profile == "causal",
        required_figures=required_figures,
    )
    return failures


def _audit_status(audit_dir: Path, results_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_AUDIT_ARTIFACTS if not (audit_dir / name).exists()]
    summary = _read_json(audit_dir / "human_audit_summary.json")
    manifest = _read_json(audit_dir / "audit_manifest.json")
    failures = []
    if summary:
        failures.extend(
            check_human_audit_readiness(
                summary,
                min_completion_rate=1.0,
                min_label_n=1,
                required_labels=DEFAULT_REQUIRED_LABELS,
                require_baseline_deltas=True,
                allow_single_annotator=False,
            )
        )
    failures.extend(check_audit_input_source_match(manifest))
    if summary and manifest:
        failures.extend(check_audit_summary_source_match(summary, manifest))
    failures.extend(_audit_generated_output_failures(audit_dir, summary, manifest))
    failures.extend(_audit_result_source_failures(manifest, results_dir))
    return {
        "path": str(audit_dir),
        "complete": not missing and not failures,
        "missing": missing,
        "failures": failures,
        "manifest_present": bool(manifest),
        "expected_audit_count": summary.get("expected_audit_count") if summary else None,
        "completed_audit_count": summary.get("completed_audit_count") if summary else None,
        "completion_rate": summary.get("completion_rate") if summary else None,
        "annotation_source_type": summary.get("annotation_source_type") if summary else None,
    }


def _claim_status(
    path: Path,
    *,
    primary_results_dir: Path,
    causal_results_dir: Path,
    primary_audit_dir: Path,
    causal_audit_dir: Path,
) -> dict[str, Any]:
    assessment = _read_json(path)
    failures = _claim_failures(
        assessment,
        {
            "primary_metrics": primary_results_dir / "metrics.json",
            "primary_manifest": primary_results_dir / "manifest.json",
            "causal_metrics": causal_results_dir / "metrics.json",
            "causal_manifest": causal_results_dir / "manifest.json",
            "primary_audit_summary": primary_audit_dir / "human_audit_summary.json",
            "primary_audit_manifest": primary_audit_dir / "audit_manifest.json",
            "causal_audit_summary": causal_audit_dir / "human_audit_summary.json",
            "causal_audit_manifest": causal_audit_dir / "audit_manifest.json",
        },
    )
    failures.extend(_claim_generated_output_failures(path, assessment))
    return {
        "path": str(path),
        "exists": path.exists(),
        "passed": bool(assessment) and not failures,
        "failures": failures,
        "passed_claim_count": assessment.get("passed_claim_count") if assessment else None,
        "recommended_framing": assessment.get("recommended_framing") if assessment else None,
        "human_audit_required": (assessment.get("human_audit_support") or {}).get("required")
        if assessment
        else None,
        "human_audit_passed": (assessment.get("human_audit_support") or {}).get("passed")
        if assessment
        else None,
    }


def _pdf_status(
    path: Path, *, expected_sources: list[tuple[str, Path]] | None = None
) -> dict[str, Any]:
    failure = _pdf_failure(path)
    text_failure = "" if failure else _pdf_text_failure(path)
    if text_failure:
        failure = text_failure
    provenance_failure = (
        "" if failure else _pdf_provenance_failure(path, expected_sources=expected_sources)
    )
    if provenance_failure:
        failure = provenance_failure
    return {
        "path": str(path),
        "exists": path.exists(),
        "valid": path.exists() and not failure,
        "failure": failure,
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": file_sha256(path),
        "provenance_manifest": str(_pdf_manifest_path(path)),
    }


def _pdf_failure(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        content = path.read_bytes()
    except OSError as exc:
        return str(exc)
    if not content.startswith(b"%PDF-"):
        return "missing PDF signature"
    if len(content) < 32:
        return "PDF too small"
    if b"%%EOF" not in content[-2048:]:
        return "missing PDF EOF marker"
    return _pdf_page_failure(path)


def _pdf_text_failure(path: Path) -> str:
    try:
        text, extractor = extract_pdf_text(path)
    except Exception as exc:
        return f"PDF text extraction failed: {exc}"
    failures = final_pdf_text_failures(text, extractor)
    if failures:
        return f"{extractor}: " + "; ".join(failures)
    return ""


def _pdf_provenance_failure(
    path: Path, *, expected_sources: list[tuple[str, Path]] | None = None
) -> str:
    if path.name != FINAL_PDF_NAME:
        return ""
    manifest_path = _pdf_manifest_path(path)
    manifest = _read_json(manifest_path)
    if not manifest:
        return f"missing final PDF provenance manifest: {manifest_path}"
    failures = []
    if manifest.get("schema_version") != 1:
        failures.append("invalid_pdf_manifest_schema")
    pdf_entry = manifest.get("pdf")
    if not isinstance(pdf_entry, dict):
        failures.append("missing_pdf_manifest_pdf_entry")
    elif pdf_entry.get("sha256") != file_sha256(path):
        failures.append("stale_pdf_manifest_pdf_hash")
    source_rows = manifest.get("source_artifacts")
    if not isinstance(source_rows, list) or not source_rows:
        failures.append("missing_pdf_manifest_sources")
        source_rows = []
    observed_kinds = set()
    for idx, row in enumerate(source_rows):
        if not isinstance(row, dict):
            failures.append(f"malformed_pdf_manifest_source:{idx}")
            continue
        kind = str(row.get("kind") or "")
        observed_kinds.add(kind)
        source_path = Path(str(row.get("path") or ""))
        if not kind:
            failures.append(f"pdf_manifest_source_missing_kind:{idx}")
        if not source_path.exists():
            failures.append(f"pdf_manifest_source_missing:{kind}")
            continue
        if row.get("sha256") != file_sha256(source_path):
            failures.append(f"pdf_manifest_source_hash_stale:{kind}")
    for prefix in FINAL_PDF_REQUIRED_SOURCE_PREFIXES:
        if not any(kind.startswith(prefix) for kind in observed_kinds):
            failures.append(f"pdf_manifest_missing_source_kind:{prefix}")
    if expected_sources is not None:
        provenance_by_kind: dict[str, list[dict[str, Any]]] = {}
        for row in source_rows:
            if isinstance(row, dict):
                provenance_by_kind.setdefault(str(row.get("kind") or ""), []).append(row)
        active_manifest_paths: dict[str, Path] = {}
        for kind, expected_path in expected_sources:
            matching_rows = provenance_by_kind.get(kind, [])
            if not matching_rows:
                failures.append(f"pdf_manifest_missing_expected_source:{kind}")
                continue
            resolved_expected = expected_path.resolve()
            if not any(_manifest_path_matches(row, resolved_expected) for row in matching_rows):
                failures.append(f"pdf_manifest_unexpected_source_path:{kind}")
            if expected_path.name in {"active_asset_manifest.json", "active_audit_manifest.json"}:
                active_manifest_paths[kind] = expected_path
        expected_active_context = _expected_active_manifest_contexts(expected_sources)
        for kind, manifest_path in sorted(active_manifest_paths.items()):
            failures.extend(
                _active_copy_manifest_failures(
                    kind,
                    manifest_path,
                    expected_active_context.get(kind, {}),
                )
            )
    return "; ".join(failures)


def _pdf_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _manifest_path_matches(row: dict[str, Any], expected_path: Path) -> bool:
    try:
        observed = Path(str(row.get("path") or "")).resolve()
    except OSError:
        return False
    return observed == expected_path


def _active_copy_manifest_failures(
    kind: str,
    manifest_path: Path,
    expected_context: dict[str, Any] | None = None,
) -> list[str]:
    expected_context = expected_context or {}
    manifest = _read_json(manifest_path)
    if not manifest:
        return [f"{kind}:missing_or_invalid_active_manifest"]
    failures: list[str] = []
    if manifest.get("schema_version") != 2:
        failures.append(f"{kind}:invalid_active_manifest_schema")
    for manifest_field, expected_path in expected_context.get("top_level_paths", {}).items():
        observed = Path(str(manifest.get(manifest_field) or ""))
        if observed.resolve() != expected_path.resolve():
            failures.append(f"{kind}:active_manifest_unexpected_{manifest_field}")
    copied = manifest.get("copied")
    if not isinstance(copied, list) or not copied:
        failures.append(f"{kind}:active_manifest_no_copied_files")
        copied = []
    missing = manifest.get("missing")
    if missing:
        failures.append(f"{kind}:active_manifest_has_missing_sources")
    active_dir = Path(str(manifest.get("active_dir") or ""))
    for idx, row in enumerate(copied):
        if not isinstance(row, dict):
            failures.append(f"{kind}:malformed_active_manifest_row:{idx}")
            continue
        source = Path(str(row.get("source") or ""))
        target = Path(str(row.get("target") or ""))
        expected_source = expected_context.get("source_by_target", {}).get(target.resolve())
        if expected_source is not None and source.resolve() != expected_source.resolve():
            failures.append(f"{kind}:active_manifest_unexpected_source_for_target:{idx}")
        if not source.exists():
            failures.append(f"{kind}:active_manifest_source_missing:{idx}")
            continue
        if not target.exists():
            failures.append(f"{kind}:active_manifest_target_missing:{idx}")
            continue
        if active_dir:
            try:
                target.resolve().relative_to(active_dir.resolve())
            except ValueError:
                failures.append(f"{kind}:active_manifest_target_outside_active_dir:{idx}")
        source_sha256 = file_sha256(source)
        target_sha256 = file_sha256(target)
        source_bytes = source.stat().st_size
        target_bytes = target.stat().st_size
        if source_sha256 != target_sha256:
            failures.append(f"{kind}:active_manifest_source_target_hash_mismatch:{idx}")
        expected_fields = {
            "source_sha256": source_sha256,
            "target_sha256": target_sha256,
            "source_bytes": source_bytes,
            "target_bytes": target_bytes,
            "sha256": target_sha256,
            "bytes": target_bytes,
        }
        for field, expected_value in expected_fields.items():
            if row.get(field) != expected_value:
                failures.append(f"{kind}:active_manifest_stale_{field}:{idx}")
    return failures


def _expected_active_manifest_contexts(
    expected_sources: list[tuple[str, Path]]
) -> dict[str, dict[str, Any]]:
    by_kind: dict[str, list[Path]] = {}
    for kind, path in expected_sources:
        by_kind.setdefault(kind, []).append(path)

    def one(kind: str) -> Path | None:
        values = by_kind.get(kind, [])
        return values[0] if values else None

    contexts: dict[str, dict[str, Any]] = {}
    active_primary_manifest = one("active_primary_manifest")
    if active_primary_manifest is not None:
        source_by_target: dict[Path, Path] = _source_by_target(
            [
                ("active_primary_main_table", "primary_generated_main_table"),
                ("active_primary_suite_table", "primary_generated_suite_table"),
                ("active_primary_macros", "primary_generated_macros"),
            ],
            by_kind,
        )
        source_by_target.update(
            _figure_source_by_target(
                by_kind.get("active_primary_figure", []),
                by_kind.get("primary_figure", []),
            )
        )
        contexts["active_primary_manifest"] = {
            "top_level_paths": {
                "generated_dir": one("primary_generated_manifest").parent
                if one("primary_generated_manifest")
                else Path(),
                "results_dir": one("primary_results_manifest").parent
                if one("primary_results_manifest")
                else Path(),
                "active_dir": active_primary_manifest.parent,
            },
            "source_by_target": source_by_target,
        }

    active_causal_manifest = one("active_causal_manifest")
    if active_causal_manifest is not None:
        source_by_target = _source_by_target(
            [
                ("active_causal_table", "causal_generated_table"),
                ("active_causal_macros", "causal_generated_macros"),
            ],
            by_kind,
        )
        source_by_target.update(
            _figure_source_by_target(
                by_kind.get("active_causal_figure", []),
                by_kind.get("causal_figure", []),
            )
        )
        contexts["active_causal_manifest"] = {
            "top_level_paths": {
                "generated_dir": one("causal_generated_manifest").parent
                if one("causal_generated_manifest")
                else Path(),
                "results_dir": one("causal_results_manifest").parent
                if one("causal_results_manifest")
                else Path(),
                "active_dir": active_causal_manifest.parent,
            },
            "source_by_target": source_by_target,
        }

    active_primary_audit_manifest = one("active_primary_audit_manifest")
    if active_primary_audit_manifest is not None:
        audit_dir = one("primary_audit_manifest").parent if one("primary_audit_manifest") else Path()
        contexts["active_primary_audit_manifest"] = {
            "top_level_paths": {
                "audit_dir": audit_dir,
                "active_dir": active_primary_audit_manifest.parent,
            },
            "source_by_target": _audit_source_by_target(
                active_primary_audit_manifest.parent,
                audit_dir,
            ),
        }

    active_causal_audit_manifest = one("active_causal_audit_manifest")
    if active_causal_audit_manifest is not None:
        audit_dir = one("causal_audit_manifest").parent if one("causal_audit_manifest") else Path()
        contexts["active_causal_audit_manifest"] = {
            "top_level_paths": {
                "audit_dir": audit_dir,
                "active_dir": active_causal_audit_manifest.parent,
            },
            "source_by_target": _audit_source_by_target(
                active_causal_audit_manifest.parent,
                audit_dir,
            ),
        }

    return contexts


def _source_by_target(
    pairs: list[tuple[str, str]], by_kind: dict[str, list[Path]]
) -> dict[Path, Path]:
    mapping = {}
    for target_kind, source_kind in pairs:
        targets = by_kind.get(target_kind, [])
        sources = by_kind.get(source_kind, [])
        if targets and sources:
            mapping[targets[0].resolve()] = sources[0]
    return mapping


def _figure_source_by_target(active_paths: list[Path], source_paths: list[Path]) -> dict[Path, Path]:
    sources_by_name = {path.name: path for path in source_paths}
    return {
        active_path.resolve(): sources_by_name[active_path.name]
        for active_path in active_paths
        if active_path.name in sources_by_name
    }


def _audit_source_by_target(active_dir: Path, audit_dir: Path) -> dict[Path, Path]:
    return {
        (active_dir / name).resolve(): audit_dir / name
        for name in [
            "audit_manifest.json",
            "human_audit_summary.json",
            "human_audit_summary.md",
            "human_audit_summary_table.tex",
            "human_audit_deltas_table.tex",
        ]
    }


def _final_pdf_expected_sources(
    *,
    primary_results_dir: Path,
    causal_results_dir: Path,
    primary_audit_dir: Path,
    causal_audit_dir: Path,
    primary_generated_dir: Path,
    causal_generated_dir: Path,
    claim_assessment_path: Path,
) -> list[tuple[str, Path]]:
    claim_generated_dir = claim_assessment_path.parent
    active_primary_dir = primary_generated_dir.parent / "active_primary"
    active_causal_dir = causal_generated_dir.parent / "active_causal"
    active_primary_audit_dir = primary_audit_dir.parent / "active_primary_summary"
    active_causal_audit_dir = causal_audit_dir.parent / "active_causal_summary"
    return [
        ("latex_main", Path("paper/latex/main.tex")),
        ("bibliography", Path("paper/references.bib")),
        ("primary_results_manifest", primary_results_dir / "manifest.json"),
        ("primary_results_metrics", primary_results_dir / "metrics.json"),
        ("primary_figures_manifest", primary_results_dir / "figures" / "manifest.json"),
        ("causal_results_manifest", causal_results_dir / "manifest.json"),
        ("causal_results_metrics", causal_results_dir / "metrics.json"),
        ("causal_figures_manifest", causal_results_dir / "figures" / "manifest.json"),
        ("primary_generated_manifest", primary_generated_dir / "artifact_manifest.json"),
        ("primary_generated_main_table", primary_generated_dir / "main_results_table.tex"),
        ("primary_generated_suite_table", primary_generated_dir / "suite_level_effects_table.tex"),
        ("primary_generated_macros", primary_generated_dir / "result_macros.tex"),
        ("active_primary_manifest", active_primary_dir / "active_asset_manifest.json"),
        ("active_primary_main_table", active_primary_dir / "main_results_table.tex"),
        ("active_primary_suite_table", active_primary_dir / "suite_level_effects_table.tex"),
        ("active_primary_macros", active_primary_dir / "result_macros.tex"),
        ("causal_generated_manifest", causal_generated_dir / "artifact_manifest.json"),
        ("causal_generated_table", causal_generated_dir / "causal_restoration_table.tex"),
        ("causal_generated_macros", causal_generated_dir / "result_macros.tex"),
        ("active_causal_manifest", active_causal_dir / "active_asset_manifest.json"),
        ("active_causal_table", active_causal_dir / "causal_restoration_table.tex"),
        ("active_causal_macros", active_causal_dir / "result_macros.tex"),
        ("claim_assessment_json", claim_assessment_path),
        ("claim_generated_status", claim_generated_dir / "abstract_status_sentence.tex"),
        ("claim_generated_table", claim_generated_dir / "claim_assessment_table.tex"),
        ("claim_generated_interpretation", claim_generated_dir / "claim_interpretation.tex"),
        ("primary_audit_manifest", primary_audit_dir / "audit_manifest.json"),
        ("primary_audit_summary_table", primary_audit_dir / "human_audit_summary_table.tex"),
        ("primary_audit_deltas_table", primary_audit_dir / "human_audit_deltas_table.tex"),
        ("active_primary_audit_manifest", active_primary_audit_dir / "active_audit_manifest.json"),
        (
            "active_primary_audit_summary_table",
            active_primary_audit_dir / "human_audit_summary_table.tex",
        ),
        (
            "active_primary_audit_deltas_table",
            active_primary_audit_dir / "human_audit_deltas_table.tex",
        ),
        ("causal_audit_manifest", causal_audit_dir / "audit_manifest.json"),
        ("causal_audit_summary_table", causal_audit_dir / "human_audit_summary_table.tex"),
        ("causal_audit_deltas_table", causal_audit_dir / "human_audit_deltas_table.tex"),
        ("active_causal_audit_manifest", active_causal_audit_dir / "active_audit_manifest.json"),
        (
            "active_causal_audit_summary_table",
            active_causal_audit_dir / "human_audit_summary_table.tex",
        ),
        (
            "active_causal_audit_deltas_table",
            active_causal_audit_dir / "human_audit_deltas_table.tex",
        ),
        ("primary_figure", primary_results_dir / "figures" / "safety_capability_phase_portrait.pdf"),
        ("primary_figure", primary_results_dir / "figures" / "selective_safety_erasure_heatmap.pdf"),
        ("primary_figure", primary_results_dir / "figures" / "prompt_effect_constellation.pdf"),
        ("primary_figure", primary_results_dir / "figures" / "cache_state_fingerprint.pdf"),
        ("primary_figure", primary_results_dir / "figures" / "safety_state_atlas.pdf"),
        ("primary_figure", primary_results_dir / "figures" / "policy_uncertainty_braid.pdf"),
        ("active_primary_figure", active_primary_dir / "figures" / "safety_capability_phase_portrait.pdf"),
        ("active_primary_figure", active_primary_dir / "figures" / "selective_safety_erasure_heatmap.pdf"),
        ("active_primary_figure", active_primary_dir / "figures" / "prompt_effect_constellation.pdf"),
        ("active_primary_figure", active_primary_dir / "figures" / "cache_state_fingerprint.pdf"),
        ("active_primary_figure", active_primary_dir / "figures" / "safety_state_atlas.pdf"),
        ("active_primary_figure", active_primary_dir / "figures" / "policy_uncertainty_braid.pdf"),
        ("causal_figure", causal_results_dir / "figures" / "causal_restoration_fraction.pdf"),
        ("causal_figure", causal_results_dir / "figures" / "causal_restoration_flow.pdf"),
        ("active_causal_figure", active_causal_dir / "figures" / "causal_restoration_fraction.pdf"),
        ("active_causal_figure", active_causal_dir / "figures" / "causal_restoration_flow.pdf"),
    ]


def _arxiv_status(
    source_dir: Path,
    archive: Path,
    *,
    primary_generated_dir: Path = Path("paper/generated/h200_qwen_full_sweep"),
    causal_generated_dir: Path = Path("paper/generated/h200_causal_patch_qwen7b"),
    primary_audit_dir: Path = Path("paper/audit/h200_qwen_full_sweep_summary"),
    causal_audit_dir: Path = Path("paper/audit/h200_causal_patch_qwen7b_summary"),
) -> dict[str, Any]:
    manifest_path = source_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    failures = []
    if not source_dir.exists():
        failures.append("missing_source_dir")
    if not manifest_path.exists():
        failures.append("missing_manifest")
    if not archive.exists():
        failures.append("missing_archive")
    if manifest:
        if manifest.get("schema_version") != 1:
            failures.append("invalid_manifest_schema")
        if manifest.get("allow_missing"):
            failures.append("allow_missing_enabled")
        for key in [
            "missing_figures",
            "invalid_figures",
            "missing_generated",
            "invalid_generated",
            "missing_audit",
            "invalid_audit",
        ]:
            if manifest.get(key):
                failures.append(key)
        for source_name, manifest_key in [
            ("main.tex", "main_tex_sha256"),
            ("references.bib", "references_sha256"),
        ]:
            source_path = source_dir / source_name
            if not source_path.exists():
                failures.append(f"missing_source_file:{source_name}")
            elif manifest.get(manifest_key) != file_sha256(source_path):
                failures.append(f"stale_source_hash:{source_name}")
        main_tex_path = source_dir / "main.tex"
        if main_tex_path.exists():
            main_tex = main_tex_path.read_text(encoding="utf-8")
            for marker in _rewrite_failures(main_tex):
                failures.append(f"main_tex_repo_local_path:{marker}")
            for failure in _final_source_failures(main_tex):
                failures.append(f"main_tex_draft_placeholder:{failure}")
        copied_generated_names = {
            Path(str(path)).name for path in manifest.get("copied_generated", [])
        }
        for required_name in [
            primary_generated_dir.name,
            causal_generated_dir.name,
            "active_primary",
            "active_causal",
            "claim_assessment",
        ]:
            if required_name not in copied_generated_names:
                failures.append(f"missing_required_generated:{required_name}")
        copied_audit_names = {
            Path(str(path)).name for path in manifest.get("copied_audit", [])
        }
        for required_name in [
            primary_audit_dir.name,
            causal_audit_dir.name,
            "active_primary_summary",
            "active_causal_summary",
        ]:
            if required_name not in copied_audit_names:
                failures.append(f"missing_required_audit:{required_name}")
        for required_figure in REQUIRED_ARXIV_FIGURE_FILES:
            figure_path = source_dir / required_figure
            if not figure_path.exists():
                continue
            figure_failure = _pdf_failure(figure_path)
            if figure_failure:
                failures.append(
                    f"invalid_required_figure_pdf:{required_figure}:{figure_failure}"
                )
        for copied_path in manifest.get("copied_figures", []):
            figure_path = _resolve_bundle_path(source_dir, copied_path)
            if not figure_path.exists():
                continue
            figure_failure = _pdf_failure(figure_path)
            if figure_failure:
                failures.append(f"invalid_copied_figure_pdf:{copied_path}:{figure_failure}")
        for key in ["copied_figures", "copied_generated", "copied_audit"]:
            for copied_path in manifest.get(key, []):
                if not _resolve_bundle_path(source_dir, copied_path).exists():
                    failures.append(f"missing_copied_path:{copied_path}")
        required_bundle_files = _required_arxiv_bundle_files(
            primary_generated_dir=primary_generated_dir,
            causal_generated_dir=causal_generated_dir,
            primary_audit_dir=primary_audit_dir,
            causal_audit_dir=causal_audit_dir,
        )
        for required_file in [*required_bundle_files, *REQUIRED_ARXIV_FIGURE_FILES]:
            if not (source_dir / required_file).exists():
                failures.append(f"missing_required_bundle_file:{required_file}")
        failures.extend(_arxiv_support_content_failures(source_dir, required_bundle_files))
        provenance_members = _manifest_provenance_members(source_dir, manifest, failures)
        for required_file in [
            "main.tex",
            "references.bib",
            *required_bundle_files,
            *REQUIRED_ARXIV_FIGURE_FILES,
        ]:
            if required_file not in provenance_members:
                failures.append(f"missing_provenance_for_required_bundle_file:{required_file}")
        failures.extend(_copied_file_provenance_failures(source_dir, manifest))
    if archive.exists():
        if archive.stat().st_size <= 0:
            failures.append("empty_archive")
        archive_hashes, archive_error, archive_failures = _archive_hashes(archive)
        failures.extend(archive_failures)
        if archive_error:
            failures.append(f"invalid_archive:{archive_error}")
        else:
            for member in ["main.tex", "references.bib", "manifest.json"]:
                if member not in archive_hashes:
                    failures.append(f"archive_missing:{member}")
                    continue
                source_member = source_dir / member
                if (
                    source_member.exists()
                    and archive_hashes[member] != file_sha256(source_member)
                ):
                    failures.append(f"archive_stale:{member}")
            if manifest:
                copied_files = _manifest_provenance_files(source_dir, manifest, failures)
                if not copied_files:
                    copied_files = _manifest_copied_files(source_dir, manifest, failures)
                expected_archive_members = {
                    "main.tex",
                    "references.bib",
                    "manifest.json",
                    *[
                        source_path.relative_to(source_dir).as_posix()
                        for source_path in copied_files
                    ],
                }
                for source_path in copied_files:
                    member = source_path.relative_to(source_dir).as_posix()
                    archive_sha = archive_hashes.get(member)
                    if archive_sha is None:
                        failures.append(f"archive_missing:{member}")
                    elif archive_sha != file_sha256(source_path):
                        failures.append(f"archive_stale:{member}")
                for member in archive_hashes:
                    if _is_raw_evidence_archive_member(member):
                        failures.append(f"archive_raw_evidence_file:{member}")
                    if _is_empirical_archive_member(member) and member not in expected_archive_members:
                        failures.append(f"archive_unmanifested_empirical_file:{member}")
    return {
        "source_dir": str(source_dir),
        "archive": str(archive),
        "complete": not failures,
        "failures": failures,
        "manifest_present": bool(manifest),
        "archive_exists": archive.exists(),
        "archive_sha256": file_sha256(archive),
    }


def _arxiv_support_content_failures(
    source_dir: Path, required_bundle_files: list[str]
) -> list[str]:
    candidates = []
    for required_file in required_bundle_files:
        path = source_dir / required_file
        if path.suffix == ".tex" and path.exists():
            candidates.append(path)
    for root in [source_dir / "generated", source_dir / "audit"]:
        if root.exists():
            candidates.extend(sorted(path for path in root.rglob("*.tex") if path.is_file()))
    return [
        f"arxiv_support_content:{failure}"
        for failure in _invalid_arxiv_support_files(sorted(set(candidates)))
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _archive_hashes(archive: Path) -> tuple[dict[str, str], str, list[str]]:
    try:
        with tarfile.open(archive, "r:gz") as tar:
            hashes = {}
            failures = []
            seen_members = set()
            for member in tar.getmembers():
                if member.name in seen_members:
                    failures.append(f"archive_duplicate:{member.name}")
                seen_members.add(member.name)
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    failures.append(f"archive_unsafe_member:{member.name}")
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                hashes[member.name] = _sha256_bytes(extracted.read())
            return hashes, "", failures
    except (tarfile.TarError, OSError) as exc:
        return {}, str(exc), []


def _manifest_copied_files(
    source_dir: Path, manifest: dict[str, Any], failures: list[str]
) -> list[Path]:
    files: list[Path] = []
    for key in ["copied_figures", "copied_generated", "copied_audit"]:
        for raw_path in manifest.get(key, []):
            path = _resolve_bundle_path(source_dir, raw_path)
            try:
                path.resolve().relative_to(source_dir.resolve())
            except ValueError:
                failures.append(f"copied_path_outside_source:{raw_path}")
                continue
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(child for child in path.rglob("*") if child.is_file()))
            else:
                failures.append(f"missing_copied_path:{raw_path}")
    return files


def _manifest_provenance_files(
    source_dir: Path, manifest: dict[str, Any], failures: list[str]
) -> list[Path]:
    provenance = manifest.get("copied_file_provenance")
    if not isinstance(provenance, list):
        return []
    files = []
    for idx, row in enumerate(provenance):
        if not isinstance(row, dict):
            failures.append(f"malformed_copied_file_provenance:{idx}")
            continue
        bundle_path = _resolve_bundle_path(source_dir, row.get("bundle_path", ""))
        try:
            bundle_path.resolve().relative_to(source_dir.resolve())
        except ValueError:
            failures.append(f"provenance_bundle_outside_source:{row.get('bundle_path') or idx}")
            continue
        if bundle_path.is_file():
            files.append(bundle_path)
        else:
            failures.append(f"provenance_bundle_missing:{row.get('bundle_path') or idx}")
    return files


def _manifest_provenance_members(
    source_dir: Path, manifest: dict[str, Any], failures: list[str]
) -> set[str]:
    provenance = manifest.get("copied_file_provenance")
    if not isinstance(provenance, list):
        return set()
    members = set()
    for idx, row in enumerate(provenance):
        if not isinstance(row, dict):
            failures.append(f"malformed_copied_file_provenance:{idx}")
            continue
        bundle_path = _resolve_bundle_path(source_dir, row.get("bundle_path", ""))
        try:
            members.add(bundle_path.resolve().relative_to(source_dir.resolve()).as_posix())
        except ValueError:
            failures.append(f"provenance_bundle_outside_source:{row.get('bundle_path') or idx}")
    return members


def _is_empirical_archive_member(member: str) -> bool:
    return member.startswith(("figures/", "generated/", "audit/"))


def _is_raw_evidence_archive_member(member: str) -> bool:
    path = Path(member)
    return path.name in RAW_EVIDENCE_BASENAMES or path.suffix in RAW_EVIDENCE_SUFFIXES


def _copied_file_provenance_failures(source_dir: Path, manifest: dict[str, Any]) -> list[str]:
    provenance = manifest.get("copied_file_provenance")
    if not isinstance(provenance, list) or not provenance:
        return ["missing_copied_file_provenance"]
    failures = []
    for idx, row in enumerate(provenance):
        if not isinstance(row, dict):
            failures.append(f"malformed_copied_file_provenance:{idx}")
            continue
        source_path = Path(str(row.get("source_path", "")))
        bundle_path = _resolve_bundle_path(source_dir, row.get("bundle_path", ""))
        label = str(row.get("bundle_path") or idx)
        kind = str(row.get("kind") or "")
        if not source_path.exists():
            failures.append(f"provenance_source_missing:{label}")
            continue
        if not bundle_path.exists():
            failures.append(f"provenance_bundle_missing:{label}")
            continue
        try:
            bundle_path.resolve().relative_to(source_dir.resolve())
        except ValueError:
            failures.append(f"provenance_bundle_outside_source:{label}")
        if kind in {"figure", "generated", "audit"} or label.startswith(
            ("figures/", "generated/", "audit/")
        ):
            try:
                source_path.resolve().relative_to(source_dir.resolve())
            except ValueError:
                pass
            else:
                failures.append(f"provenance_source_self_referential:{label}")
        source_sha = file_sha256(source_path)
        bundle_sha = file_sha256(bundle_path)
        if row.get("source_sha256") != source_sha:
            failures.append(f"provenance_source_hash_stale:{label}")
        if row.get("bundle_sha256") != bundle_sha:
            failures.append(f"provenance_bundle_hash_stale:{label}")
        if row.get("direct_copy") is not False and source_sha != bundle_sha:
            failures.append(f"provenance_direct_copy_mismatch:{label}")
    return failures


def _resolve_bundle_path(source_dir: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return source_dir / path


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _audit_result_source_failures(manifest: dict[str, Any], results_dir: Path) -> list[str]:
    if not manifest:
        return []
    result_sources = (manifest.get("source_artifacts") or {}).get("results")
    if not isinstance(result_sources, dict):
        return ["missing_result_source_manifest"]
    failures = []
    for name in ["manifest.json", "generations.jsonl", "metrics.json"]:
        source = result_sources.get(name)
        if not isinstance(source, dict):
            failures.append(f"missing_result_source:{name}")
            continue
        path = results_dir / name
        if not path.exists():
            failures.append(f"missing_result_artifact:{name}")
            continue
        if source.get("sha256") != file_sha256(path):
            failures.append(f"stale_result_source:{name}")
    return failures


def _claim_failures(assessment: dict[str, Any], source_paths: dict[str, Path]) -> list[str]:
    if not assessment:
        return []
    failures = []
    if assessment.get("schema_version") != 1:
        failures.append("invalid_claim_schema")
    thresholds = assessment.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        failures.append("missing_claim_thresholds")
    if not str(assessment.get("recommended_framing") or "").strip():
        failures.append("missing_recommended_framing")
    claims = assessment.get("claims")
    if not isinstance(claims, dict):
        failures.append("missing_claims")
    else:
        for claim_id in EXPECTED_CLAIM_IDS:
            claim = claims.get(claim_id)
            if not isinstance(claim, dict):
                failures.append(f"missing_claim:{claim_id}")
                continue
            if claim.get("passed") is not True:
                failures.append(f"claim_failed:{claim_id}")
            evidence_count = _claim_evidence_count(claim_id, claim)
            if evidence_count <= 0:
                failures.append(f"claim_lacks_evidence:{claim_id}")
            failures.extend(_claim_best_evidence_failures(claim_id, claim))
    if assessment.get("passed_claim_count") != len(EXPECTED_CLAIM_IDS):
        failures.append(
            f"passed_claim_count={assessment.get('passed_claim_count')}; "
            f"expected={len(EXPECTED_CLAIM_IDS)}"
        )
    publication_gate = assessment.get("publication_gate")
    if not isinstance(publication_gate, dict):
        failures.append("missing_publication_gate")
    elif publication_gate.get("passed") is not True:
        failures.append("publication_gate_failed")
    if isinstance(publication_gate, dict):
        required_claims = publication_gate.get("required_claims")
        if not isinstance(required_claims, list):
            failures.append("publication_gate_lacks_required_claims")
        else:
            required_claim_set = {str(claim) for claim in required_claims}
            for claim_id in EXPECTED_PUBLICATION_REQUIRED_CLAIMS:
                if claim_id not in required_claim_set:
                    failures.append(f"publication_gate_missing_required_claim:{claim_id}")
            unexpected = sorted(required_claim_set - set(EXPECTED_PUBLICATION_REQUIRED_CLAIMS))
            for claim_id in unexpected:
                failures.append(f"publication_gate_unexpected_required_claim:{claim_id}")
    audit_support = assessment.get("human_audit_support")
    if not isinstance(audit_support, dict):
        failures.append("missing_human_audit_support")
    else:
        if audit_support.get("required") is not True:
            failures.append("human_audit_support_not_required")
        if audit_support.get("passed") is not True:
            failures.append("human_audit_support_failed")
        if not isinstance(audit_support.get("best_primary_delta"), dict):
            failures.append("human_audit_lacks_primary_delta")
        causal_delta = audit_support.get("best_causal_delta") or audit_support.get(
            "best_causal_restoration_delta"
        )
        if not isinstance(causal_delta, dict):
            failures.append("human_audit_lacks_causal_restoration_delta")
    failures.extend(_claim_source_failures(assessment, source_paths))
    failures.extend(_claim_recompute_failures(assessment, source_paths))
    return failures


def _claim_evidence_count(claim_id: str, claim: dict[str, Any]) -> int:
    if claim_id == "H3_causal_safety_state_erasure":
        raw_count = claim.get("eligible_comparison_count")
    else:
        raw_count = claim.get("eligible_evidence_count")
    try:
        return int(raw_count)
    except (TypeError, ValueError):
        return 0


def _claim_best_evidence_failures(claim_id: str, claim: dict[str, Any]) -> list[str]:
    if claim_id == "H3_causal_safety_state_erasure":
        comparison = claim.get("best_comparison")
        if not isinstance(comparison, dict):
            return [f"claim_lacks_best_comparison:{claim_id}"]
        failures = []
        for key in ["margin", "margin_ci_low", "system_ci_width", "control_ci_width", "passed"]:
            if key not in comparison:
                failures.append(f"claim_lacks_best_comparison_{key}:{claim_id}")
        for key in ["system_patch", "matched_user_control"]:
            evidence = comparison.get(key)
            if not isinstance(evidence, dict):
                failures.append(f"claim_lacks_best_comparison_{key}:{claim_id}")
                continue
            for evidence_key in ["key", "metric", "value", "ci_low", "ci_high"]:
                if evidence_key not in evidence:
                    failures.append(
                        f"claim_lacks_best_comparison_{key}_{evidence_key}:{claim_id}"
                    )
        return failures
    evidence = claim.get("best_evidence")
    if not isinstance(evidence, dict):
        return [f"claim_lacks_best_evidence:{claim_id}"]
    return [
        f"claim_lacks_best_evidence_{key}:{claim_id}"
        for key in ["key", "estimate", "ci_low", "ci_high", "ci_width", "passed"]
        if key not in evidence
    ]


def _claim_recompute_failures(
    assessment: dict[str, Any], source_paths: dict[str, Path]
) -> list[str]:
    required_sources = [
        "primary_metrics",
        "causal_metrics",
        "primary_audit_summary",
        "causal_audit_summary",
    ]
    missing = [name for name in required_sources if not source_paths[name].exists()]
    if missing:
        return [f"claim_recompute_missing_source:{name}" for name in missing]
    thresholds = assessment.get("thresholds") if isinstance(assessment, dict) else {}
    if not isinstance(thresholds, dict):
        thresholds = {}
    recomputed = assess_claims(
        _read_json(source_paths["primary_metrics"]),
        _read_json(source_paths["causal_metrics"]),
        primary_audit_metrics=_read_json(source_paths["primary_audit_summary"]),
        causal_audit_metrics=_read_json(source_paths["causal_audit_summary"]),
        min_safety_effect=float(thresholds.get("min_safety_effect_ci_low", 0.02)),
        min_ssei_effect=float(thresholds.get("min_ssei_effect_ci_low", 0.02)),
        min_restoration_fraction=float(thresholds.get("min_restoration_fraction", 0.20)),
        min_restoration_margin=float(
            thresholds.get("min_restoration_margin_over_user_control", 0.10)
        ),
        min_human_audit_delta=float(thresholds.get("min_human_audit_delta", 0.0)),
        max_primary_ci_width=float(thresholds.get("max_primary_ci_width", PRIMARY_MAX_CI_WIDTH)),
        max_causal_ci_width=float(thresholds.get("max_causal_ci_width", CAUSAL_MAX_CI_WIDTH)),
        require_human_audit_support=True,
    )
    failures = []
    if recomputed.get("publication_gate", {}).get("passed") is not True:
        failures.append("claim_recompute_publication_gate_failed")
    if recomputed.get("passed_claim_count") != assessment.get("passed_claim_count"):
        failures.append(
            "claim_recompute_passed_count_mismatch:"
            f"{recomputed.get('passed_claim_count')}!={assessment.get('passed_claim_count')}"
        )
    recomputed_claims = recomputed.get("claims") or {}
    assessment_claims = assessment.get("claims") or {}
    for claim_id in EXPECTED_CLAIM_IDS:
        recomputed_passed = (recomputed_claims.get(claim_id) or {}).get("passed")
        assessment_passed = (assessment_claims.get(claim_id) or {}).get("passed")
        if recomputed_passed is not assessment_passed:
            failures.append(
                f"claim_recompute_pass_mismatch:{claim_id}:"
                f"{recomputed_passed}!={assessment_passed}"
            )
    if recomputed.get("human_audit_support", {}).get("passed") is not (
        assessment.get("human_audit_support") or {}
    ).get("passed"):
        failures.append("claim_recompute_human_audit_mismatch")
    return failures


def _audit_generated_output_failures(
    audit_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    if not summary or not audit_dir.name.endswith("_summary"):
        return []
    expected = {
        "human_audit_summary.md": render_summary_markdown(summary),
        "human_audit_summary_table.tex": render_summary_latex(summary),
        "human_audit_deltas_table.tex": render_deltas_latex(summary),
    }
    failures = []
    for name, expected_text in expected.items():
        path = audit_dir / name
        if not path.exists():
            failures.append(f"missing_audit_generated_output:{name}")
            continue
        observed = path.read_text(encoding="utf-8", errors="replace")
        if observed != expected_text:
            failures.append(f"stale_audit_generated_output:{name}")
    generated_manifest = manifest.get("generated_artifacts") if isinstance(manifest, dict) else None
    if not isinstance(generated_manifest, dict):
        failures.append("missing_audit_generated_artifact_manifest")
    else:
        for name in expected:
            source = generated_manifest.get(name)
            path = audit_dir / name
            if not isinstance(source, dict):
                failures.append(f"missing_audit_generated_artifact:{name}")
            elif path.exists() and source.get("sha256") != file_sha256(path):
                failures.append(f"stale_audit_generated_artifact:{name}")
    return failures


def _claim_generated_output_failures(path: Path, assessment: dict[str, Any]) -> list[str]:
    if not assessment or path.parent.name != "claim_assessment":
        return []
    expected = {
        "claim_assessment_table.tex": render_latex_table(assessment),
        "claim_interpretation.tex": render_interpretation_latex(assessment),
        "abstract_status_sentence.tex": render_abstract_status_latex(assessment),
    }
    failures = []
    for name, expected_text in expected.items():
        output_path = path.parent / name
        if not output_path.exists():
            failures.append(f"missing_claim_generated_output:{name}")
            continue
        observed = output_path.read_text(encoding="utf-8", errors="replace")
        if observed != expected_text:
            failures.append(f"stale_claim_generated_output:{name}")
    artifact_manifest = _read_json(path.parent / "artifact_manifest.json")
    generated_manifest = artifact_manifest.get("generated_artifacts")
    if not isinstance(generated_manifest, dict):
        failures.append("missing_claim_artifact_manifest")
    else:
        for name in ["claim_assessment.json", *expected]:
            source = generated_manifest.get(name)
            output_path = path.parent / name
            if not isinstance(source, dict):
                failures.append(f"missing_claim_generated_artifact:{name}")
            elif output_path.exists() and source.get("sha256") != file_sha256(output_path):
                failures.append(f"stale_claim_generated_artifact:{name}")
    return failures


def _claim_source_failures(assessment: dict[str, Any], source_paths: dict[str, Path]) -> list[str]:
    source_artifacts = assessment.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        return ["missing_claim_source_artifacts"]
    failures = []
    for name, path in source_paths.items():
        source = source_artifacts.get(name)
        if not isinstance(source, dict):
            failures.append(f"missing_claim_source:{name}")
            continue
        if not path.exists():
            failures.append(f"missing_claim_source_file:{name}")
            continue
        if source.get("sha256") != file_sha256(path):
            failures.append(f"stale_claim_source:{name}")
    return failures


def _artifact_line(label: str, status: dict[str, Any]) -> str:
    state = "complete" if status["complete"] else "blocked"
    details = []
    if status.get("missing"):
        details.append(f"missing {len(status['missing'])}")
    if status.get("disqualifiers"):
        details.append("disqualified: " + ", ".join(status["disqualifiers"]))
    if status.get("failures"):
        details.append("failed: " + ", ".join(status["failures"]))
    if status.get("readiness_failures"):
        details.append("readiness: " + ", ".join(status["readiness_failures"]))
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"- {label}: `{state}` at `{status['path']}`{suffix}"


def _audit_artifact_label(prefix: str, status: dict[str, Any]) -> str:
    source_type = str(status.get("annotation_source_type") or "human").strip().lower()
    if source_type == "open_local_judge":
        return f"{prefix} open local judge audit"
    if source_type == "mixed":
        return f"{prefix} mixed-source audit"
    return f"{prefix} human audit"


def _claim_line(status: dict[str, Any]) -> str:
    state = "pass" if status["passed"] else "blocked"
    suffix = ""
    if status.get("failures"):
        suffix = " (failed: " + ", ".join(status["failures"]) + ")"
    return f"- claim assessment: `{state}` at `{status['path']}`{suffix}"


def _pdf_line(status: dict[str, Any], *, evidence_ready: bool) -> str:
    if not status["exists"]:
        state = "missing"
        suffix = ""
    elif not status.get("valid"):
        state = "invalid"
        suffix = f" ({status.get('failure', 'invalid PDF')})"
    elif evidence_ready:
        state = "exists"
        suffix = ""
    else:
        state = "draft-only"
        suffix = " (evidence gates incomplete; not a publishable paper)"
    return f"- paper PDF: `{state}` at `{status['path']}`{suffix}"


def _arxiv_line(status: dict[str, Any]) -> str:
    if status["complete"]:
        state = "complete"
    elif status.get("archive_exists"):
        state = "stale"
    else:
        state = "blocked"
    suffix = ""
    if status.get("failures"):
        suffix = " (failed: " + ", ".join(status["failures"]) + ")"
    return f"- arXiv bundle: `{state}` at `{status['archive']}`{suffix}"


if __name__ == "__main__":
    main()
