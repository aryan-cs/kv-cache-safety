from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from assess_claims import assess_claims
from check_human_audit_readiness import (
    DEFAULT_REQUIRED_LABELS,
    check_audit_input_source_match,
    check_human_audit_readiness,
)
from check_publication_readiness import _check_figure_manifest
from package_arxiv_submission import FIGURE_SOURCES, _rewrite_failures

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
REQUIRED_ARXIV_BUNDLE_FILES = [
    "generated/h200_qwen_full_sweep/main_results_table.tex",
    "generated/h200_qwen_full_sweep/suite_level_effects_table.tex",
    "generated/h200_qwen_full_sweep/result_macros.tex",
    "generated/h200_causal_patch_qwen7b/causal_restoration_table.tex",
    "generated/h200_causal_patch_qwen7b/result_macros.tex",
    "generated/claim_assessment/abstract_status_sentence.tex",
    "generated/claim_assessment/claim_assessment_table.tex",
    "generated/claim_assessment/claim_interpretation.tex",
    "audit/h200_qwen_full_sweep_summary/human_audit_summary_table.tex",
    "audit/h200_qwen_full_sweep_summary/human_audit_deltas_table.tex",
    "audit/h200_causal_patch_qwen7b_summary/human_audit_summary_table.tex",
    "audit/h200_causal_patch_qwen7b_summary/human_audit_deltas_table.tex",
]
REQUIRED_ARXIV_FIGURE_FILES = [f"figures/{name}" for name in FIGURE_SOURCES]
PRIMARY_REQUIRED_FIGURES = [
    "safety_capability_phase_portrait",
    "selective_safety_erasure_heatmap",
    "prompt_effect_constellation",
    "cache_state_fingerprint",
    "safety_state_atlas",
]
CAUSAL_REQUIRED_FIGURES = [
    "causal_restoration_fraction",
    "causal_restoration_flow",
]
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
    pdf = _pdf_status(paper_pdf)
    arxiv = _arxiv_status(arxiv_source_dir, arxiv_archive)

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
            _artifact_line("primary human audit", status["primary_human_audit"]),
            _artifact_line("causal human audit", status["causal_human_audit"]),
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
    failures.extend(_prompt_suite_provenance_failures(results_dir, manifest))
    failures.extend(_figure_source_failures(results_dir))
    failures.extend(_figure_manifest_failures(results_dir, profile=profile))
    return failures


def _jsonl_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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
    if public_without_provenance:
        failures.append(f"public_prompts_lack_dataset_provenance:{public_without_provenance}")
    return failures


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


def _pdf_status(path: Path) -> dict[str, Any]:
    failure = _pdf_failure(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "valid": path.exists() and not failure,
        "failure": failure,
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": file_sha256(path),
    }


def _pdf_failure(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        prefix = path.read_bytes()[:5]
    except OSError as exc:
        return str(exc)
    if prefix != b"%PDF-":
        return "missing PDF signature"
    return ""


def _arxiv_status(source_dir: Path, archive: Path) -> dict[str, Any]:
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
        for key in ["missing_figures", "invalid_figures", "missing_generated", "missing_audit"]:
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
            for marker in _rewrite_failures(main_tex_path.read_text(encoding="utf-8")):
                failures.append(f"main_tex_repo_local_path:{marker}")
        copied_generated_names = {
            Path(str(path)).name for path in manifest.get("copied_generated", [])
        }
        for required_name in [
            "h200_qwen_full_sweep",
            "h200_causal_patch_qwen7b",
            "claim_assessment",
        ]:
            if required_name not in copied_generated_names:
                failures.append(f"missing_required_generated:{required_name}")
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
        for required_file in [*REQUIRED_ARXIV_BUNDLE_FILES, *REQUIRED_ARXIV_FIGURE_FILES]:
            if not (source_dir / required_file).exists():
                failures.append(f"missing_required_bundle_file:{required_file}")
        provenance_members = _manifest_provenance_members(source_dir, manifest, failures)
        for required_file in [
            "main.tex",
            "references.bib",
            *REQUIRED_ARXIV_BUNDLE_FILES,
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
        for key in ["system_patch", "matched_user_control"]:
            if not isinstance(comparison.get(key), dict):
                failures.append(f"claim_lacks_best_comparison_{key}:{claim_id}")
        return failures
    if not isinstance(claim.get("best_evidence"), dict):
        return [f"claim_lacks_best_evidence:{claim_id}"]
    return []


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
