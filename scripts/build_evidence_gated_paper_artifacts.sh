#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
primary_generated_dir="${PRIMARY_GENERATED_DIR:-paper/generated/h200_qwen_full_sweep}"
causal_generated_dir="${CAUSAL_GENERATED_DIR:-paper/generated/h200_causal_patch_qwen7b}"
claim_generated_dir="${CLAIM_GENERATED_DIR:-paper/generated/claim_assessment}"
primary_audit_summary="${PRIMARY_AUDIT_SUMMARY_DIR:-paper/audit/h200_qwen_full_sweep_summary}"
causal_audit_summary="${CAUSAL_AUDIT_SUMMARY_DIR:-paper/audit/h200_causal_patch_qwen7b_summary}"
publication_status_dir="${PUBLICATION_STATUS_DIR:-paper/build}"
arxiv_source_dir="${ARXIV_SOURCE_DIR:-paper/build/arxiv_source}"
arxiv_archive="${ARXIV_ARCHIVE:-paper/build/arxiv_source.tar.gz}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required evidence-gated paper artifact: $path" >&2
    exit 1
  fi
}

clear_stale_publication_pdfs() {
  rm -f \
    paper/cache_mediated_safety_erasure.pdf \
    paper/cache_mediated_safety_erasure.pdf.manifest.json \
    paper/build/cache_mediated_safety_erasure.pdf \
    paper/build/cache_mediated_safety_erasure.pdf.manifest.json
}

for path in \
  "$primary_results/manifest.json" \
  "$primary_results/metrics.json" \
  "$primary_results/figures/manifest.json" \
  "$causal_results/manifest.json" \
  "$causal_results/metrics.json" \
  "$causal_results/figures/manifest.json" \
  "$primary_generated_dir/artifact_manifest.json" \
  "$causal_generated_dir/artifact_manifest.json" \
  "$claim_generated_dir/claim_assessment.json" \
  "$claim_generated_dir/abstract_status_sentence.tex" \
  "$claim_generated_dir/claim_assessment_table.tex" \
  "$claim_generated_dir/claim_interpretation.tex" \
  "$primary_audit_summary/audit_manifest.json" \
  "$primary_audit_summary/human_audit_summary_table.tex" \
  "$primary_audit_summary/human_audit_deltas_table.tex" \
  "$causal_audit_summary/audit_manifest.json" \
  "$causal_audit_summary/human_audit_summary_table.tex" \
  "$causal_audit_summary/human_audit_deltas_table.tex"; do
  require_file "$path"
done

clear_stale_publication_pdfs

uv run python scripts/sync_active_paper_assets.py \
  --primary-results-dir "$primary_results" \
  --causal-results-dir "$causal_results" \
  --primary-generated-dir "$primary_generated_dir" \
  --causal-generated-dir "$causal_generated_dir" \
  --primary-audit-dir "$primary_audit_summary" \
  --causal-audit-dir "$causal_audit_summary" \
  --strict

uv run python scripts/check_latex_placeholders.py --tex paper/latex/main.tex
uv run python scripts/check_paper_asset_freshness.py \
  --pair "$primary_generated_dir=$primary_results" \
  --pair "$causal_generated_dir=$causal_results" \
  --require-exported-table-set \
  --require-recomputed-output

PRIMARY_RESULTS_DIR="$primary_results" \
CAUSAL_RESULTS_DIR="$causal_results" \
PRIMARY_PAPER_DIR="$primary_generated_dir" \
CAUSAL_PAPER_DIR="$causal_generated_dir" \
PRIMARY_AUDIT_SUMMARY_DIR="$primary_audit_summary" \
CAUSAL_AUDIT_SUMMARY_DIR="$causal_audit_summary" \
CLAIM_ASSESSMENT_PATH="$claim_generated_dir/claim_assessment.json" \
ARXIV_SOURCE_DIR="$arxiv_source_dir" \
ARXIV_ARCHIVE="$arxiv_archive" \
REQUIRE_COMPLETE_PAPER=1 \
ALLOW_NONPASSING_CLAIM_PAPER=1 \
bash scripts/build_paper_pdf.sh

uv run python scripts/package_arxiv_submission.py \
  --output-dir "$arxiv_source_dir" \
  --archive "$arxiv_archive" \
  --primary-results-dir "$primary_results" \
  --causal-results-dir "$causal_results" \
  --primary-generated-dir "$primary_generated_dir" \
  --causal-generated-dir "$causal_generated_dir" \
  --claim-generated-dir "$claim_generated_dir" \
  --primary-audit-dir "$primary_audit_summary" \
  --causal-audit-dir "$causal_audit_summary"

mkdir -p "$publication_status_dir"
uv run python scripts/report_publication_status.py \
  --primary-results-dir "$primary_results" \
  --causal-results-dir "$causal_results" \
  --primary-audit-dir "$primary_audit_summary" \
  --causal-audit-dir "$causal_audit_summary" \
  --claim-assessment "$claim_generated_dir/claim_assessment.json" \
  --primary-generated-dir "$primary_generated_dir" \
  --causal-generated-dir "$causal_generated_dir" \
  --arxiv-source-dir "$arxiv_source_dir" \
  --arxiv-archive "$arxiv_archive" \
  --require-arxiv-bundle \
  --output-json "$publication_status_dir/publication_status.json" \
  --output-md "$publication_status_dir/publication_status.md"

echo "Evidence-gated paper artifacts rebuilt:"
echo "- paper/cache_mediated_safety_erasure.pdf"
echo "- $arxiv_archive"
echo "- $publication_status_dir/publication_status.md"
