#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

src_dir="paper/latex"
build_dir="paper/build"
primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
primary_paper_dir="${PRIMARY_PAPER_DIR:-paper/generated/h200_qwen_full_sweep}"
causal_paper_dir="${CAUSAL_PAPER_DIR:-paper/generated/h200_causal_patch_qwen7b}"
active_primary_dir="paper/generated/active_primary"
active_causal_dir="paper/generated/active_causal"
primary_audit_dir="${PRIMARY_AUDIT_SUMMARY_DIR:-paper/audit/h200_qwen_full_sweep_summary}"
causal_audit_dir="${CAUSAL_AUDIT_SUMMARY_DIR:-paper/audit/h200_causal_patch_qwen7b_summary}"
active_primary_audit_dir="paper/audit/active_primary_summary"
active_causal_audit_dir="paper/audit/active_causal_summary"
claim_assessment="${CLAIM_ASSESSMENT_PATH:-paper/generated/claim_assessment/claim_assessment.json}"
arxiv_source_dir="${ARXIV_SOURCE_DIR:-paper/build/arxiv_source}"
arxiv_archive="${ARXIV_ARCHIVE:-paper/build/arxiv_source.tar.gz}"
branch="${BRANCH:-master}"
mkdir -p "$build_dir"

require_current_origin_head() {
  git fetch origin "$branch"
  local head
  local origin_head
  head="$(git rev-parse HEAD)"
  origin_head="$(git rev-parse "origin/$branch")"
  if [[ "$head" != "$origin_head" ]]; then
    echo "Refusing to build a complete paper PDF from a stale checkout: HEAD=${head}, origin/${branch}=${origin_head}." >&2
    echo "Fetch or fast-forward to origin/${branch} before regenerating final paper assets." >&2
    exit 1
  fi
}

if [[ "${REQUIRE_COMPLETE_PAPER:-0}" == "1" ]]; then
  if [[ -n "$(git status --short)" ]]; then
    echo "Refusing to build a complete paper PDF from a dirty git working tree." >&2
    git status --short >&2
    exit 1
  fi
  require_current_origin_head
fi

final_pdf_sources=(
  "latex_main=$src_dir/main.tex"
  "bibliography=paper/references.bib"
  "primary_results_manifest=$primary_results/manifest.json"
  "primary_results_metrics=$primary_results/metrics.json"
  "primary_figures_manifest=$primary_results/figures/manifest.json"
  "causal_results_manifest=$causal_results/manifest.json"
  "causal_results_metrics=$causal_results/metrics.json"
  "causal_figures_manifest=$causal_results/figures/manifest.json"
  "primary_generated_manifest=$primary_paper_dir/artifact_manifest.json"
  "primary_generated_main_table=$primary_paper_dir/main_results_table.tex"
  "primary_generated_suite_table=$primary_paper_dir/suite_level_effects_table.tex"
  "primary_generated_macros=$primary_paper_dir/result_macros.tex"
  "active_primary_manifest=$active_primary_dir/active_asset_manifest.json"
  "active_primary_main_table=$active_primary_dir/main_results_table.tex"
  "active_primary_suite_table=$active_primary_dir/suite_level_effects_table.tex"
  "active_primary_macros=$active_primary_dir/result_macros.tex"
  "causal_generated_manifest=$causal_paper_dir/artifact_manifest.json"
  "causal_generated_table=$causal_paper_dir/causal_restoration_table.tex"
  "causal_generated_macros=$causal_paper_dir/result_macros.tex"
  "active_causal_manifest=$active_causal_dir/active_asset_manifest.json"
  "active_causal_table=$active_causal_dir/causal_restoration_table.tex"
  "active_causal_macros=$active_causal_dir/result_macros.tex"
  "claim_assessment_json=$claim_assessment"
  "claim_generated_status=$(dirname "$claim_assessment")/abstract_status_sentence.tex"
  "claim_generated_table=$(dirname "$claim_assessment")/claim_assessment_table.tex"
  "claim_generated_interpretation=$(dirname "$claim_assessment")/claim_interpretation.tex"
  "primary_audit_manifest=$primary_audit_dir/audit_manifest.json"
  "primary_audit_summary_table=$primary_audit_dir/human_audit_summary_table.tex"
  "primary_audit_deltas_table=$primary_audit_dir/human_audit_deltas_table.tex"
  "active_primary_audit_manifest=$active_primary_audit_dir/active_audit_manifest.json"
  "active_primary_audit_summary_table=$active_primary_audit_dir/human_audit_summary_table.tex"
  "active_primary_audit_deltas_table=$active_primary_audit_dir/human_audit_deltas_table.tex"
  "causal_audit_manifest=$causal_audit_dir/audit_manifest.json"
  "causal_audit_summary_table=$causal_audit_dir/human_audit_summary_table.tex"
  "causal_audit_deltas_table=$causal_audit_dir/human_audit_deltas_table.tex"
  "active_causal_audit_manifest=$active_causal_audit_dir/active_audit_manifest.json"
  "active_causal_audit_summary_table=$active_causal_audit_dir/human_audit_summary_table.tex"
  "active_causal_audit_deltas_table=$active_causal_audit_dir/human_audit_deltas_table.tex"
  "primary_figure=$primary_results/figures/safety_capability_phase_portrait.pdf"
  "primary_figure=$primary_results/figures/selective_safety_erasure_heatmap.pdf"
  "primary_figure=$primary_results/figures/prompt_effect_constellation.pdf"
  "primary_figure=$primary_results/figures/cache_state_fingerprint.pdf"
  "primary_figure=$primary_results/figures/safety_state_atlas.pdf"
  "primary_figure=$primary_results/figures/policy_uncertainty_braid.pdf"
  "active_primary_figure=$active_primary_dir/figures/safety_capability_phase_portrait.pdf"
  "active_primary_figure=$active_primary_dir/figures/selective_safety_erasure_heatmap.pdf"
  "active_primary_figure=$active_primary_dir/figures/prompt_effect_constellation.pdf"
  "active_primary_figure=$active_primary_dir/figures/cache_state_fingerprint.pdf"
  "active_primary_figure=$active_primary_dir/figures/safety_state_atlas.pdf"
  "active_primary_figure=$active_primary_dir/figures/policy_uncertainty_braid.pdf"
  "causal_figure=$causal_results/figures/causal_restoration_fraction.pdf"
  "causal_figure=$causal_results/figures/causal_restoration_flow.pdf"
  "active_causal_figure=$active_causal_dir/figures/causal_restoration_fraction.pdf"
  "active_causal_figure=$active_causal_dir/figures/causal_restoration_flow.pdf"
)

publication_status_args=(
  --primary-results-dir "$primary_results"
  --causal-results-dir "$causal_results"
  --primary-audit-dir "$primary_audit_dir"
  --causal-audit-dir "$causal_audit_dir"
  --claim-assessment "$claim_assessment"
  --primary-generated-dir "$primary_paper_dir"
  --causal-generated-dir "$causal_paper_dir"
  --arxiv-source-dir "$arxiv_source_dir"
  --arxiv-archive "$arxiv_archive"
)
publication_status_fail_args=()
if [[ "${ALLOW_NONPASSING_CLAIM_PAPER:-0}" != "1" ]]; then
  publication_status_fail_args+=(--fail-if-not-ready)
fi

require_valid_pdf() {
  local pdf="$1"
  if [[ ! -s "$pdf" ]]; then
    echo "LaTeX build did not produce a nonempty PDF: $pdf" >&2
    exit 1
  fi
  if [[ "$(head -c 5 "$pdf")" != "%PDF-" ]]; then
    echo "LaTeX build produced an invalid PDF: $pdf" >&2
    exit 1
  fi
}

check_final_pdf_text() {
  local pdf="$1"
  if [[ "${ALLOW_DRAFT_PDF:-0}" == "1" ]]; then
    echo "Skipping final PDF text check because ALLOW_DRAFT_PDF=1." >&2
    return
  fi
  if ! uv run python scripts/check_final_pdf_text.py --pdf "$pdf"; then
    rm -f "$pdf" "${pdf}.manifest.json"
    exit 1
  fi
}

write_final_pdf_manifest() {
  local pdf="$1"
  local output="$2"
  local cmd=(uv run python scripts/write_final_pdf_manifest.py --pdf "$pdf" --output "$output")
  local source
  for source in "${final_pdf_sources[@]}"; do
    cmd+=(--source "$source")
  done
  "${cmd[@]}"
}

uv run python scripts/sync_active_paper_assets.py \
  --primary-results-dir "$primary_results" \
  --causal-results-dir "$causal_results" \
  --primary-generated-dir "$primary_paper_dir" \
  --causal-generated-dir "$causal_paper_dir" \
  --primary-audit-dir "$primary_audit_dir" \
  --causal-audit-dir "$causal_audit_dir" \
  --active-primary-dir "$active_primary_dir" \
  --active-causal-dir "$active_causal_dir" \
  --active-primary-audit-dir "$active_primary_audit_dir" \
  --active-causal-audit-dir "$active_causal_audit_dir" \
  --strict

if [[ "${REQUIRE_COMPLETE_PAPER:-0}" == "1" ]]; then
  uv run python scripts/check_latex_citations.py \
    --tex "$src_dir/main.tex" \
    --bib paper/references.bib \
    --require-all-bib-used
  uv run python scripts/check_latex_placeholders.py --tex "$src_dir/main.tex"
  uv run python scripts/check_paper_asset_freshness.py \
    --pair "$primary_paper_dir=$primary_results" \
    --pair "$causal_paper_dir=$causal_results" \
    --require-exported-table-set \
    --require-recomputed-output \
    --require-current-analysis-commit
  uv run python scripts/report_publication_status.py \
    "${publication_status_args[@]}" \
    --paper-pdf "$build_dir/cache_mediated_safety_erasure.pdf" \
    --allow-missing-paper-pdf \
    "${publication_status_fail_args[@]}"
fi

rm -f "$build_dir/main.pdf" "$build_dir/cache_mediated_safety_erasure.pdf"
rm -f \
  "$build_dir/main.aux" \
  "$build_dir/main.bbl" \
  "$build_dir/main.bcf" \
  "$build_dir/main.blg" \
  "$build_dir/main.fdb_latexmk" \
  "$build_dir/main.fls" \
  "$build_dir/main.log" \
  "$build_dir/main.out" \
  "$build_dir/main.run.xml" \
  "$build_dir/main.synctex.gz" \
  "$build_dir/main.toc"

if command -v tectonic >/dev/null 2>&1; then
  (
    cd "$src_dir"
    tectonic --outdir ../build main.tex
  )
elif command -v latexmk >/dev/null 2>&1; then
  (
    cd "$src_dir"
    latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
  )
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  (
    cd "$src_dir"
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
    bibtex ../build/main
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
  )
else
  echo "No supported LaTeX builder found. Install tectonic, latexmk, or pdflatex+bibtex." >&2
  exit 1
fi

require_valid_pdf "$build_dir/main.pdf"
mv "$build_dir/main.pdf" "$build_dir/cache_mediated_safety_erasure.pdf"
require_valid_pdf "$build_dir/cache_mediated_safety_erasure.pdf"
check_final_pdf_text "$build_dir/cache_mediated_safety_erasure.pdf"
write_final_pdf_manifest \
  "$build_dir/cache_mediated_safety_erasure.pdf" \
  "$build_dir/cache_mediated_safety_erasure.pdf.manifest.json"

if [[ "${REQUIRE_COMPLETE_PAPER:-0}" == "1" ]]; then
  uv run python scripts/report_publication_status.py \
    "${publication_status_args[@]}" \
    --paper-pdf "$build_dir/cache_mediated_safety_erasure.pdf" \
    "${publication_status_fail_args[@]}"
fi

cp "$build_dir/cache_mediated_safety_erasure.pdf" paper/cache_mediated_safety_erasure.pdf
require_valid_pdf paper/cache_mediated_safety_erasure.pdf
check_final_pdf_text paper/cache_mediated_safety_erasure.pdf
write_final_pdf_manifest \
  "paper/cache_mediated_safety_erasure.pdf" \
  "paper/cache_mediated_safety_erasure.pdf.manifest.json"

echo "Wrote $build_dir/cache_mediated_safety_erasure.pdf"
echo "Wrote paper/cache_mediated_safety_erasure.pdf"
