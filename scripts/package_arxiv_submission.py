from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json

FIGURE_SOURCES = {
    "safety_capability_phase_portrait.pdf": Path(
        "results/h200_qwen_full_sweep/figures/safety_capability_phase_portrait.pdf"
    ),
    "selective_safety_erasure_heatmap.pdf": Path(
        "results/h200_qwen_full_sweep/figures/selective_safety_erasure_heatmap.pdf"
    ),
    "prompt_effect_constellation.pdf": Path(
        "results/h200_qwen_full_sweep/figures/prompt_effect_constellation.pdf"
    ),
    "cache_state_fingerprint.pdf": Path(
        "results/h200_qwen_full_sweep/figures/cache_state_fingerprint.pdf"
    ),
    "safety_state_atlas.pdf": Path(
        "results/h200_qwen_full_sweep/figures/safety_state_atlas.pdf"
    ),
    "causal_restoration_fraction.pdf": Path(
        "results/h200_causal_patch_qwen7b/figures/causal_restoration_fraction.pdf"
    ),
    "causal_restoration_flow.pdf": Path(
        "results/h200_causal_patch_qwen7b/figures/causal_restoration_flow.pdf"
    ),
}
REQUIRED_GENERATED_DIRS = [
    Path("paper/generated/h200_qwen_full_sweep"),
    Path("paper/generated/h200_causal_patch_qwen7b"),
    Path("paper/generated/claim_assessment"),
]
OPTIONAL_GENERATED_DIRS = [
    Path("paper/generated/h200_qwen32b_public_followup"),
]
GENERATED_DIRS = REQUIRED_GENERATED_DIRS + OPTIONAL_GENERATED_DIRS
AUDIT_DIRS = [
    Path("paper/audit/h200_qwen_full_sweep_summary"),
    Path("paper/audit/h200_causal_patch_qwen7b_summary"),
]
ARXIV_SAFE_SUPPORT_SUFFIXES = {".tex"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an arXiv-friendly LaTeX source bundle.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/build/arxiv_source"))
    parser.add_argument("--archive", type=Path, default=Path("paper/build/arxiv_source.tar.gz"))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing empirical assets for draft bundles. Publication builds should not use this.",
    )
    args = parser.parse_args()

    source_dir = args.output_dir
    if source_dir.exists():
        shutil.rmtree(source_dir)
    (source_dir / "figures").mkdir(parents=True)

    main_tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")
    rewritten_main_tex = _rewrite_main_tex_for_arxiv(main_tex)
    rewrite_failures = _rewrite_failures(rewritten_main_tex)
    if rewrite_failures:
        for failure in rewrite_failures:
            print(f"Unrewritten LaTeX source path: {failure}")
        raise SystemExit("Refusing to package arXiv source with repo-local paths in main.tex.")
    (source_dir / "main.tex").write_text(rewritten_main_tex, encoding="utf-8")
    shutil.copyfile("paper/references.bib", source_dir / "references.bib")
    copied_file_provenance = [
        _file_provenance(
            kind="latex_main",
            source_path=Path("paper/latex/main.tex"),
            bundle_path=source_dir / "main.tex",
            bundle_root=source_dir,
            direct_copy=False,
            transform="rewrite_main_tex_for_arxiv",
        ),
        _file_provenance(
            kind="bibliography",
            source_path=Path("paper/references.bib"),
            bundle_path=source_dir / "references.bib",
            bundle_root=source_dir,
        ),
    ]

    copied_figures = []
    missing_figures = []
    invalid_figures = []
    for output_name, source_path in FIGURE_SOURCES.items():
        target_path = source_dir / "figures" / output_name
        if source_path.exists():
            if not _is_pdf(source_path):
                invalid_figures.append(str(source_path))
                continue
            shutil.copyfile(source_path, target_path)
            copied_figures.append(_bundle_manifest_path(source_dir, target_path))
            copied_file_provenance.append(
                _file_provenance(
                    kind="figure",
                    source_path=source_path,
                    bundle_path=target_path,
                    bundle_root=source_dir,
                )
            )
        else:
            missing_figures.append(str(source_path))
    copied_generated = []
    missing_generated = []
    for source_path in REQUIRED_GENERATED_DIRS:
        if not source_path.exists():
            missing_generated.append(str(source_path))
            continue
        target_path = source_dir / "generated" / source_path.name
        copied_files = _copy_arxiv_support_tree(source_path, target_path)
        copied_generated.append(_bundle_manifest_path(source_dir, target_path))
        copied_file_provenance.extend(
            _directory_provenance("generated", source_path, target_path, copied_files)
        )
    skipped_optional_generated = []
    for source_path in OPTIONAL_GENERATED_DIRS:
        if not source_path.exists():
            skipped_optional_generated.append(str(source_path))
            continue
        target_path = source_dir / "generated" / source_path.name
        copied_files = _copy_arxiv_support_tree(source_path, target_path)
        copied_generated.append(_bundle_manifest_path(source_dir, target_path))
        copied_file_provenance.extend(
            _directory_provenance("generated", source_path, target_path, copied_files)
        )
    copied_audit = []
    missing_audit = []
    for source_path in AUDIT_DIRS:
        if not source_path.exists():
            missing_audit.append(str(source_path))
            continue
        target_path = source_dir / "audit" / source_path.name
        copied_files = _copy_arxiv_support_tree(source_path, target_path)
        copied_audit.append(_bundle_manifest_path(source_dir, target_path))
        copied_file_provenance.extend(
            _directory_provenance("audit", source_path, target_path, copied_files)
        )

    manifest = {
        "schema_version": 1,
        "main_tex_sha256": file_sha256(source_dir / "main.tex"),
        "references_sha256": file_sha256(source_dir / "references.bib"),
        "copied_figures": copied_figures,
        "missing_figures": missing_figures,
        "invalid_figures": invalid_figures,
        "copied_generated": copied_generated,
        "missing_generated": missing_generated,
        "skipped_optional_generated": skipped_optional_generated,
        "copied_audit": copied_audit,
        "missing_audit": missing_audit,
        "copied_file_provenance": copied_file_provenance,
        "allow_missing": args.allow_missing,
    }
    write_json(source_dir / "manifest.json", manifest)
    if invalid_figures:
        for path in invalid_figures:
            print(f"Invalid required arXiv figure PDF: {path}")
        raise SystemExit("Refusing to package arXiv bundle with invalid figure PDFs.")
    missing_inputs = _missing_inputs(manifest)
    if missing_inputs and not args.allow_missing:
        for path in missing_inputs:
            print(f"Missing required arXiv input: {path}")
        raise SystemExit(
            "Refusing to package publication arXiv bundle with missing empirical inputs. "
            "Use --allow-missing only for draft/pre-results bundles."
        )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        for path in sorted(source_dir.rglob("*")):
            archive.add(path, arcname=path.relative_to(source_dir))
    print(f"Wrote {args.archive}")


def _rewrite_main_tex_for_arxiv(text: str) -> str:
    text = text.replace(r"\bibliography{../references}", r"\bibliography{references}")
    text = text.replace("../generated/", "generated/")
    text = text.replace("../audit/", "audit/")
    for output_name, source_path in FIGURE_SOURCES.items():
        text = text.replace(str(Path("../..") / source_path), f"figures/{output_name}")
    return text


def _rewrite_failures(text: str) -> list[str]:
    markers = [
        "../generated/",
        "../audit/",
        "../references",
        "../../results/",
        "/Users/aryan/Desktop/projects/llm-safety",
        "/home/aryang9/sandbox/llm-safety",
    ]
    return [marker for marker in markers if marker in text]


def _missing_inputs(manifest: dict) -> list[str]:
    return [
        *manifest.get("missing_figures", []),
        *manifest.get("missing_generated", []),
        *manifest.get("missing_audit", []),
    ]


def _is_pdf(path: Path) -> bool:
    try:
        return path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def _copy_arxiv_support_tree(source_dir: Path, bundle_dir: Path) -> list[Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for source_file in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        if source_file.suffix not in ARXIV_SAFE_SUPPORT_SUFFIXES:
            continue
        bundle_file = bundle_dir / source_file.relative_to(source_dir)
        bundle_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, bundle_file)
        copied_files.append(source_file)
    return copied_files


def _directory_provenance(
    kind: str, source_dir: Path, bundle_dir: Path, copied_files: list[Path] | None = None
) -> list[dict[str, object]]:
    rows = []
    source_files = copied_files
    if source_files is None:
        source_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    for source_file in source_files:
        bundle_file = bundle_dir / source_file.relative_to(source_dir)
        rows.append(
            _file_provenance(
                kind=kind,
                source_path=source_file,
                bundle_path=bundle_file,
                bundle_root=bundle_dir.parent.parent,
            )
        )
    return rows


def _file_provenance(
    *,
    kind: str,
    source_path: Path,
    bundle_path: Path,
    bundle_root: Path | None = None,
    direct_copy: bool = True,
    transform: str | None = None,
) -> dict[str, object]:
    manifest_bundle_path = (
        _bundle_manifest_path(bundle_root, bundle_path) if bundle_root is not None else str(bundle_path)
    )
    row: dict[str, object] = {
        "kind": kind,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_bytes": source_path.stat().st_size if source_path.exists() else None,
        "bundle_path": manifest_bundle_path,
        "bundle_sha256": file_sha256(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size if bundle_path.exists() else None,
        "direct_copy": direct_copy,
    }
    if transform is not None:
        row["transform"] = transform
    return row


def _bundle_manifest_path(source_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(source_dir.resolve()).as_posix()


if __name__ == "__main__":
    main()
