import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from report_publication_status import publication_status, render_markdown

REQUIRED_AUDIT_LABELS = [
    "human_refusal_correct",
    "human_unsafe_compliance",
    "human_system_leakage",
]


def test_publication_status_reports_missing_artifacts_as_blockers(tmp_path: Path) -> None:
    status = publication_status(
        primary_results_dir=tmp_path / "primary",
        causal_results_dir=tmp_path / "causal",
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=tmp_path / "paper.pdf",
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert "claim_assessment_passed" in status["blockers"]
    assert "paper_pdf_exists" in status["blockers"]
    assert "paper_pdf_valid" in status["blockers"]


def test_publication_status_can_ignore_pdf_when_prechecking_complete_build(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=tmp_path / "missing.pdf",
        require_paper_pdf=False,
    )

    assert status["publication_ready"] is True
    assert status["paper_pdf_required"] is False
    assert "paper_pdf_exists" not in status["blockers"]
    assert "paper_pdf_valid" not in status["blockers"]


def test_publication_markdown_marks_existing_pdf_as_draft_until_evidence_ready(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=tmp_path / "primary",
        causal_results_dir=tmp_path / "causal",
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=pdf_path,
    )
    rendered = render_markdown(status)

    assert status["evidence_ready"] is False
    assert "paper PDF: `draft-only`" in rendered
    assert "evidence gates incomplete" in rendered


def test_publication_status_accepts_complete_real_artifacts(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is True
    assert status["blockers"] == []
    assert "Publication ready: `true`" in render_markdown(status)


def test_publication_status_rejects_non_pdf_final_paper(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_text("not a real pdf", encoding="utf-8")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "paper_pdf_valid" in status["blockers"]
    assert status["paper_pdf"]["failure"] == "missing PDF signature"
    assert "paper PDF: `invalid`" in render_markdown(status)


def test_publication_status_requires_complete_arxiv_bundle_when_requested(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is True
    assert status["gates"]["arxiv_bundle_ready"] is True


def test_publication_status_rejects_malformed_arxiv_figure_pdf(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive, valid_figure_pdf=False)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert status["gates"]["arxiv_bundle_ready"] is False
    assert any(
        failure.startswith("invalid_copied_figure_pdf:")
        for failure in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_stale_arxiv_provenance_source(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    copied_figure = arxiv_dir / "figures" / "figure.pdf"
    copied_figure.write_bytes(b"%PDF-1.7\nchanged\n")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert any(
        failure.startswith("provenance_source_hash_stale:")
        for failure in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_requires_arxiv_file_provenance(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        manifest_overrides={"copied_file_provenance": []},
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "missing_copied_file_provenance" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_arxiv_archive_missing_manifest_assets(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(
        arxiv_dir,
        archive,
        include_manifest_assets_in_archive=False,
    )
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert status["gates"]["arxiv_bundle_ready"] is False
    assert "archive_missing:figures/figure.pdf" in status["arxiv_bundle"]["failures"]
    assert (
        "archive_missing:generated/h200_qwen_full_sweep/main_results_table.tex"
        in status["arxiv_bundle"]["failures"]
    )


def test_publication_status_rejects_unsafe_or_duplicate_archive_members(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive)
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(arxiv_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(arxiv_dir))
        tar.add(arxiv_dir / "main.tex", arcname="main.tex")
        unsafe = tarfile.TarInfo("../escape.tex")
        payload = b"unsafe\n"
        unsafe.size = len(payload)
        tar.addfile(unsafe, io.BytesIO(payload))
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "archive_duplicate:main.tex" in status["arxiv_bundle"]["failures"]
    assert "archive_unsafe_member:../escape.tex" in status["arxiv_bundle"]["failures"]


def test_publication_status_rejects_draft_arxiv_bundle_when_required(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    arxiv_dir = tmp_path / "arxiv_source"
    archive = tmp_path / "arxiv_source.tar.gz"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    _write_arxiv_bundle(arxiv_dir, archive, manifest_overrides={"allow_missing": True})
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
        arxiv_source_dir=arxiv_dir,
        arxiv_archive=archive,
        require_arxiv_bundle=True,
    )

    assert status["publication_ready"] is False
    assert "arxiv_bundle_ready" in status["blockers"]
    assert "allow_missing_enabled" in status["arxiv_bundle"]["failures"]
    assert "arXiv bundle: `stale`" in render_markdown(status)


def test_publication_status_rejects_stale_audit_source_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "metrics.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert "stale_result_source:metrics.json" in status["primary_human_audit"]["failures"]


def test_publication_status_rejects_audits_without_inter_annotator_pairs(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary, include_inter_annotator=False)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert (
        "`human_refusal_correct` has no inter-annotator pairs"
        in status["primary_human_audit"]["failures"]
    )


def test_publication_status_rejects_stale_audit_input_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary_audit / "audit_labels.csv").write_text("changed\n", encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert any(
        "audit CSV source 0 hash is stale" in failure
        for failure in status["primary_human_audit"]["failures"]
    )


def test_publication_status_rejects_stale_claim_source_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim = _passing_claim_assessment(primary, causal, primary_audit, causal_audit)
    claim["source_artifacts"]["causal_metrics"]["sha256"] = "stale"
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "stale_claim_source:causal_metrics" in status["claim_assessment"]["failures"]


def test_publication_status_rejects_preliminary_claim_assessment_without_audit_gate(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(
            {
                "publication_gate": {"passed": True},
                "passed_claim_count": 3,
                "source_artifacts": _claim_source_artifacts(
                    primary, causal, primary_audit, causal_audit
                ),
                "human_audit_support": {
                    "required": False,
                    "passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "human_audit_support_not_required" in status["claim_assessment"]["failures"]


def test_publication_status_rejects_smoke_or_mock_runs(tmp_path: Path) -> None:
    primary = tmp_path / "primary_smoke"
    causal = tmp_path / "causal"
    _write_run(primary, manifest_overrides={"model_provider": "mock", "run_name": "smoke"})
    _write_run(causal)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=tmp_path / "paper.pdf",
    )

    assert status["publication_ready"] is False
    assert "mock_model" in status["primary_results"]["disqualifiers"]
    assert "smoke_run" in status["primary_results"]["disqualifiers"]


def test_publication_status_rejects_obvious_run_readiness_failures(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "generations.jsonl").write_text('{"row": 1}\n', encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment(primary, causal, primary_audit, causal_audit)),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert any(
        failure.startswith("generation_row_count=1; expected=2")
        for failure in status["primary_results"]["readiness_failures"]
    )
    assert any(
        failure == "figures_manifest_source_hash_stale:generations.jsonl"
        for failure in status["primary_results"]["readiness_failures"]
    )


def _write_run(path: Path, manifest_overrides: dict | None = None) -> None:
    (path / "figures").mkdir(parents=True)
    manifest = {
        "model_provider": "hf",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "run_name": "h200_qwen_full_sweep",
        "git_dirty": False,
        "git_commit": "abc123",
        "expected_generation_count": 2,
        "cache_policy_labels": ["none", "kv_int4_sim"],
        "cache_policy_configs": [{"name": "none"}, {"name": "kv_int4_sim"}],
        "prompt_counts": {"public_refusal_safety": 650},
    }
    manifest.update(manifest_overrides or {})
    for name in ["config.resolved.yaml", "environment.json", "prompts.jsonl", "cache_stats.parquet"]:
        (path / name).write_text("artifact\n", encoding="utf-8")
    (path / "generations.jsonl").write_text(
        '{"suite":"public_refusal_safety","prompt_id":"p1","policy":"none","seed":0}\n'
        '{"suite":"public_refusal_safety","prompt_id":"p1","policy":"kv_int4_sim","seed":0}\n',
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    figure_manifest = {
        "source_artifacts": {
            name: {"sha256": _sha256(path / name)}
            for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
        }
    }
    (path / "figures" / "manifest.json").write_text(
        json.dumps(figure_manifest),
        encoding="utf-8",
    )


def _write_audit(
    path: Path,
    results_dir: Path,
    *,
    include_inter_annotator: bool = True,
) -> None:
    path.mkdir(parents=True)
    summary = {
        "expected_audit_count": 10,
        "completed_audit_count": 10,
        "completion_rate": 1.0,
        "unknown_audit_ids": [],
        "duplicate_annotation_keys": [],
        "distinct_annotator_count": 2,
        "multi_annotator_audit_count": 2 if include_inter_annotator else 0,
        "label_rates": {label: {"n": 10, "mean": 0.5} for label in REQUIRED_AUDIT_LABELS},
        "baseline_policy_deltas": {
            f"public_refusal_safety::kv_int4_sim::{label}": {"n": 5, "mean_delta": 0.1}
            for label in REQUIRED_AUDIT_LABELS
        },
    }
    if include_inter_annotator:
        summary["inter_annotator"] = {
            label: {"pair_count": 2, "agreement": 1.0} for label in REQUIRED_AUDIT_LABELS
        }
    for name in ["human_audit_summary.md", "human_audit_summary_table.tex", "human_audit_deltas_table.tex"]:
        (path / name).write_text("artifact\n", encoding="utf-8")
    audit_csv = path / "audit_labels.csv"
    key_jsonl = path / "audit_key.jsonl"
    audit_csv.write_text("audit_id,human_refusal_correct\n1,true\n", encoding="utf-8")
    key_jsonl.write_text('{"audit_id":"1"}\n', encoding="utf-8")
    manifest = {
        "source_artifacts": {
            "audit_csv": [
                {
                    "path": str(audit_csv),
                    "sha256": _sha256(audit_csv),
                    "bytes": audit_csv.stat().st_size,
                }
            ],
            "key_jsonl": {
                "path": str(key_jsonl),
                "sha256": _sha256(key_jsonl),
                "bytes": key_jsonl.stat().st_size,
            },
            "results": {
                name: {"sha256": _sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json"]
            }
        }
    }
    (path / "audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "human_audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_arxiv_bundle(
    source_dir: Path,
    archive: Path,
    *,
    manifest_overrides: dict | None = None,
    include_manifest_assets_in_archive: bool = True,
    valid_figure_pdf: bool = True,
) -> None:
    source_dir.mkdir(parents=True)
    (source_dir / "generated" / "h200_qwen_full_sweep").mkdir(parents=True)
    (source_dir / "generated" / "h200_causal_patch_qwen7b").mkdir(parents=True)
    (source_dir / "generated" / "claim_assessment").mkdir(parents=True)
    (source_dir / "audit" / "h200_qwen_full_sweep_summary").mkdir(parents=True)
    (source_dir / "audit" / "h200_causal_patch_qwen7b_summary").mkdir(parents=True)
    (source_dir / "figures").mkdir()
    (source_dir / "main.tex").write_text("main\n", encoding="utf-8")
    (source_dir / "references.bib").write_text("refs\n", encoding="utf-8")
    figure = source_dir / "figures" / "figure.pdf"
    if valid_figure_pdf:
        figure.write_bytes(b"%PDF-1.7\n")
    else:
        figure.write_text("not a pdf\n", encoding="utf-8")
    generated_files = [
        source_dir / "generated" / "h200_qwen_full_sweep" / "main_results_table.tex",
        source_dir / "generated" / "h200_qwen_full_sweep" / "suite_level_effects_table.tex",
        source_dir / "generated" / "h200_qwen_full_sweep" / "result_macros.tex",
        source_dir / "generated" / "h200_causal_patch_qwen7b" / "causal_restoration_table.tex",
        source_dir / "generated" / "h200_causal_patch_qwen7b" / "result_macros.tex",
        source_dir / "generated" / "claim_assessment" / "abstract_status_sentence.tex",
        source_dir / "generated" / "claim_assessment" / "claim_assessment_table.tex",
        source_dir / "generated" / "claim_assessment" / "claim_interpretation.tex",
    ]
    audit_files = [
        source_dir / "audit" / "h200_qwen_full_sweep_summary" / "human_audit_summary_table.tex",
        source_dir / "audit" / "h200_qwen_full_sweep_summary" / "human_audit_deltas_table.tex",
        source_dir / "audit" / "h200_causal_patch_qwen7b_summary" / "human_audit_summary_table.tex",
        source_dir / "audit" / "h200_causal_patch_qwen7b_summary" / "human_audit_deltas_table.tex",
    ]
    for path in [*generated_files, *audit_files]:
        path.write_text(f"{path.name}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "allow_missing": False,
        "main_tex_sha256": _sha256(source_dir / "main.tex"),
        "references_sha256": _sha256(source_dir / "references.bib"),
        "copied_figures": [str(figure)],
        "copied_generated": [
            str(source_dir / "generated" / "h200_qwen_full_sweep"),
            str(source_dir / "generated" / "h200_causal_patch_qwen7b"),
            str(source_dir / "generated" / "claim_assessment"),
        ],
        "copied_audit": [
            str(source_dir / "audit" / "h200_qwen_full_sweep_summary"),
            str(source_dir / "audit" / "h200_causal_patch_qwen7b_summary"),
        ],
        "missing_figures": [],
        "missing_generated": [],
        "missing_audit": [],
    }
    manifest["copied_file_provenance"] = _arxiv_provenance_rows(
        [
            source_dir / "main.tex",
            source_dir / "references.bib",
            figure,
            *generated_files,
            *audit_files,
        ],
        source_dir,
    )
    manifest.update(manifest_overrides or {})
    (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_dir / "main.tex", arcname="main.tex")
        tar.add(source_dir / "references.bib", arcname="references.bib")
        tar.add(source_dir / "manifest.json", arcname="manifest.json")
        if include_manifest_assets_in_archive:
            tar.add(figure, arcname="figures/figure.pdf")
            for path in [*generated_files, *audit_files]:
                tar.add(path, arcname=path.relative_to(source_dir))


def _arxiv_provenance_rows(paths: list[Path], source_dir: Path) -> list[dict]:
    rows = []
    for path in paths:
        rows.append(
            {
                "kind": "test",
                "source_path": str(path),
                "source_sha256": _sha256(path),
                "source_bytes": path.stat().st_size,
                "bundle_path": str(path),
                "bundle_sha256": _sha256(path),
                "bundle_bytes": path.stat().st_size,
                "direct_copy": path.name != "main.tex",
                "relative_bundle_path": path.relative_to(source_dir).as_posix(),
            }
        )
    return rows


def _passing_claim_assessment(
    primary: Path,
    causal: Path,
    primary_audit: Path,
    causal_audit: Path,
) -> dict:
    return {
        "publication_gate": {"passed": True},
        "passed_claim_count": 3,
        "source_artifacts": _claim_source_artifacts(primary, causal, primary_audit, causal_audit),
        "human_audit_support": {
            "required": True,
            "passed": True,
        },
    }


def _claim_source_artifacts(
    primary: Path,
    causal: Path,
    primary_audit: Path,
    causal_audit: Path,
) -> dict:
    return {
        "primary_metrics": {"sha256": _sha256(primary / "metrics.json")},
        "primary_manifest": {"sha256": _sha256(primary / "manifest.json")},
        "causal_metrics": {"sha256": _sha256(causal / "metrics.json")},
        "causal_manifest": {"sha256": _sha256(causal / "manifest.json")},
        "primary_audit_summary": {
            "sha256": _sha256(primary_audit / "human_audit_summary.json")
        },
        "primary_audit_manifest": {"sha256": _sha256(primary_audit / "audit_manifest.json")},
        "causal_audit_summary": {"sha256": _sha256(causal_audit / "human_audit_summary.json")},
        "causal_audit_manifest": {"sha256": _sha256(causal_audit / "audit_manifest.json")},
    }
