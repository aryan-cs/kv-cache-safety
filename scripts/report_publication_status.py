from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from check_human_audit_readiness import (
    DEFAULT_REQUIRED_LABELS,
    check_audit_input_source_match,
    check_human_audit_readiness,
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
    primary = _run_status(primary_results_dir)
    causal = _run_status(causal_results_dir)
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


def _run_status(results_dir: Path) -> dict[str, Any]:
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
    readiness_failures = _run_readiness_failures(results_dir, manifest)
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


def _run_readiness_failures(results_dir: Path, manifest: dict[str, Any]) -> list[str]:
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
    failures.extend(_figure_source_failures(results_dir))
    return failures


def _jsonl_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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
        for key in ["missing_figures", "missing_generated", "missing_audit"]:
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
        for copied_path in manifest.get("copied_figures", []):
            figure_path = Path(str(copied_path))
            if not figure_path.exists():
                continue
            figure_failure = _pdf_failure(figure_path)
            if figure_failure:
                failures.append(f"invalid_copied_figure_pdf:{copied_path}:{figure_failure}")
        for key in ["copied_figures", "copied_generated", "copied_audit"]:
            for copied_path in manifest.get(key, []):
                if not Path(str(copied_path)).exists():
                    failures.append(f"missing_copied_path:{copied_path}")
    if archive.exists():
        if archive.stat().st_size <= 0:
            failures.append("empty_archive")
        archive_hashes, archive_error = _archive_hashes(archive)
        if archive_error:
            failures.append(f"invalid_archive:{archive_error}")
        else:
            for member in ["main.tex", "references.bib", "manifest.json"]:
                if member not in archive_hashes:
                    failures.append(f"archive_missing:{member}")
            if manifest:
                for source_path in _manifest_copied_files(source_dir, manifest, failures):
                    member = source_path.relative_to(source_dir).as_posix()
                    archive_sha = archive_hashes.get(member)
                    if archive_sha is None:
                        failures.append(f"archive_missing:{member}")
                    elif archive_sha != file_sha256(source_path):
                        failures.append(f"archive_stale:{member}")
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


def _archive_hashes(archive: Path) -> tuple[dict[str, str], str]:
    try:
        with tarfile.open(archive, "r:gz") as tar:
            hashes = {}
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                hashes[member.name] = _sha256_bytes(extracted.read())
            return hashes, ""
    except (tarfile.TarError, OSError) as exc:
        return {}, str(exc)


def _manifest_copied_files(
    source_dir: Path, manifest: dict[str, Any], failures: list[str]
) -> list[Path]:
    files: list[Path] = []
    for key in ["copied_figures", "copied_generated", "copied_audit"]:
        for raw_path in manifest.get(key, []):
            path = Path(str(raw_path))
            try:
                path.relative_to(source_dir)
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
    if not assessment.get("publication_gate", {}).get("passed"):
        failures.append("publication_gate_failed")
    audit_support = assessment.get("human_audit_support")
    if not isinstance(audit_support, dict):
        failures.append("missing_human_audit_support")
    else:
        if audit_support.get("required") is not True:
            failures.append("human_audit_support_not_required")
        if audit_support.get("passed") is not True:
            failures.append("human_audit_support_failed")
    failures.extend(_claim_source_failures(assessment, source_paths))
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
