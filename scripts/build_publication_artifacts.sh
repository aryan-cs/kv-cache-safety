#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
qwen32_results="${QWEN32_RESULTS_DIR:-results/h200_qwen32b_public_followup_primary}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
causal_ci_width="${CAUSAL_CI_WIDTH:-0.12}"
qwen32_ci_width="${QWEN32_CI_WIDTH:-0.10}"

require_result_artifacts() {
  local results_dir="$1"
  for required in manifest.json generations.jsonl metrics.json cache_stats.parquet; do
    if [[ ! -f "$results_dir/$required" ]]; then
      echo "Missing required result artifact: $results_dir/$required" >&2
      exit 1
    fi
  done
}

rebuild_primary() {
  require_result_artifacts "$primary_results"
  uv run python scripts/aggregate_results.py --results-dir "$primary_results"
  uv run python scripts/make_figures.py --results-dir "$primary_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$primary_results" \
    --paper-dir paper/generated/h200_qwen_full_sweep \
    --macro-prefix Primary
  uv run python scripts/plan_ci_power.py \
    --results-dir "$primary_results" \
    --target-ci-width "$target_ci_width" \
    --output-json "$primary_results/ci_power.json" \
    --output-md paper/generated/h200_qwen_full_sweep/ci_power.md
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$primary_results" \
    --paper-dir paper/generated/h200_qwen_full_sweep \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --suite-min-prompts public_xstest_safe=200 \
    --max-ci-width "$target_ci_width" \
    --required-suite system_leakage \
    --required-suite public_refusal_safety \
    --required-suite public_benign_overrefusal \
    --required-suite public_xstest_safe \
    --required-suite public_capability_arc \
    --required-policy none \
    --required-policy sliding_window \
    --required-policy sink_recent \
    --required-policy random_matched \
    --required-policy kv_int8_sim \
    --required-policy kv_int4_sim \
    --require-policy-pinned \
    --require-public-provenance
}

rebuild_causal() {
  require_result_artifacts "$causal_results"
  uv run python scripts/aggregate_results.py --results-dir "$causal_results"
  uv run python scripts/make_figures.py --results-dir "$causal_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$causal_results" \
    --paper-dir paper/generated/h200_causal_patch_qwen7b \
    --macro-prefix Causal
  uv run python scripts/plan_ci_power.py \
    --results-dir "$causal_results" \
    --target-ci-width "$causal_ci_width" \
    --output-json "$causal_results/ci_power.json" \
    --output-md paper/generated/h200_causal_patch_qwen7b/ci_power.md
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$causal_results" \
    --paper-dir paper/generated/h200_causal_patch_qwen7b \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --max-ci-width "$causal_ci_width" \
    --required-suite system_leakage \
    --required-suite public_refusal_safety \
    --required-policy none \
    --required-policy kv_int4_sim \
    --require-causal-patch \
    --require-policy-pinned \
    --require-public-provenance
}

rebuild_qwen32_if_present() {
  if [[ ! -d "$qwen32_results" ]]; then
    echo "Skipping Qwen 32B follow-up artifacts; directory not found: $qwen32_results"
    return
  fi
  require_result_artifacts "$qwen32_results"
  uv run python scripts/aggregate_results.py --results-dir "$qwen32_results"
  uv run python scripts/make_figures.py --results-dir "$qwen32_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$qwen32_results" \
    --paper-dir paper/generated/h200_qwen32b_public_followup \
    --macro-prefix QwenThirtyTwo
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$qwen32_results" \
    --paper-dir paper/generated/h200_qwen32b_public_followup \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --suite-min-prompts public_xstest_safe=200 \
    --max-ci-width "$qwen32_ci_width" \
    --required-suite system_leakage \
    --required-suite public_refusal_safety \
    --required-suite public_benign_overrefusal \
    --required-suite public_xstest_safe \
    --required-suite public_capability_arc \
    --required-policy none \
    --required-policy sliding_window \
    --required-policy sink_recent \
    --required-policy kv_int4_sim \
    --require-policy-pinned \
    --require-public-provenance
}

uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q

rebuild_primary
rebuild_causal
rebuild_qwen32_if_present

bash scripts/build_paper_pdf.sh
cp paper/build/cache_mediated_safety_erasure.pdf paper/cache_mediated_safety_erasure.pdf
uv run python scripts/package_arxiv_submission.py

echo "Publication artifacts rebuilt:"
echo "- paper/cache_mediated_safety_erasure.pdf"
echo "- paper/build/arxiv_source.tar.gz"
