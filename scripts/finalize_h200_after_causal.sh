#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$(pwd)/.cache/torch}"

primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
primary_generated="${PRIMARY_GENERATED_DIR:-paper/generated/h200_qwen_full_sweep}"
causal_generated="${CAUSAL_GENERATED_DIR:-paper/generated/h200_causal_patch_qwen7b}"
claim_generated="${CLAIM_GENERATED_DIR:-paper/generated/claim_assessment}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
causal_ci_width="${CAUSAL_CI_WIDTH:-0.12}"
audit_per_suite_policy="${AUDIT_PER_SUITE_POLICY:-10}"
audit_annotator_template_count="${AUDIT_ANNOTATOR_TEMPLATE_COUNT:-2}"
run_open_judge_audit="${RUN_OPEN_JUDGE_AUDIT:-0}"
attempt_publication_build="${ATTEMPT_PUBLICATION_BUILD:-1}"
staging_allow_wide_ci="${STAGING_ALLOW_WIDE_CI:-1}"

wide_ci_args=()
if [[ "$staging_allow_wide_ci" == "1" ]]; then
  wide_ci_args+=(--allow-wide-ci)
fi

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to finalize from a dirty git working tree." >&2
  git status --short >&2
  exit 1
fi

require_completed_generation_matrix() {
  local results_dir="$1"
  local expected
  local actual
  for required in config.resolved.yaml environment.json manifest.json prompts.jsonl generations.jsonl cache_stats.parquet; do
    if [[ ! -f "$results_dir/$required" ]]; then
      echo "Missing required raw result artifact: $results_dir/$required" >&2
      exit 1
    fi
  done
  expected="$(python3 - "$results_dir/manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("expected_generation_count", ""))
PY
)"
  actual="$(wc -l < "$results_dir/generations.jsonl")"
  if [[ -n "$expected" && "$expected" != "$actual" ]]; then
    echo "Incomplete generation matrix in $results_dir: actual=$actual expected=$expected" >&2
    exit 1
  fi
}

rebuild_primary() {
  require_completed_generation_matrix "$primary_results"
  uv run python scripts/aggregate_results.py --results-dir "$primary_results"
  uv run python scripts/make_figures.py --results-dir "$primary_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$primary_results" \
    --paper-dir "$primary_generated" \
    --macro-prefix Primary
  uv run python scripts/plan_ci_power.py \
    --results-dir "$primary_results" \
    --target-ci-width "$target_ci_width" \
    --output-json "$primary_results/ci_power.json" \
    --output-md "$primary_generated/ci_power.md"
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$primary_results" \
    --paper-dir "$primary_generated" \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --suite-min-prompts public_xstest_safe=200 \
    --max-ci-width "$target_ci_width" \
    "${wide_ci_args[@]}" \
    --required-suite system_leakage \
    --required-suite public_system_leakage \
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
    --required-figure safety_capability_phase_portrait \
    --required-figure selective_safety_erasure_heatmap \
    --required-figure prompt_effect_constellation \
    --required-figure cache_state_fingerprint \
    --required-figure safety_state_atlas \
    --require-public-provenance
  uv run python scripts/export_human_audit_sample.py \
    --results-dir "$primary_results" \
    --per-suite-policy "$audit_per_suite_policy" \
    --annotator-template-count "$audit_annotator_template_count" \
    --include-hidden-reference
}

rebuild_causal() {
  require_completed_generation_matrix "$causal_results"
  uv run python scripts/aggregate_results.py --results-dir "$causal_results"
  uv run python scripts/make_figures.py --results-dir "$causal_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$causal_results" \
    --paper-dir "$causal_generated" \
    --macro-prefix Causal
  uv run python scripts/plan_ci_power.py \
    --results-dir "$causal_results" \
    --target-ci-width "$causal_ci_width" \
    --output-json "$causal_results/ci_power.json" \
    --output-md "$causal_generated/ci_power.md"
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$causal_results" \
    --paper-dir "$causal_generated" \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --max-ci-width "$causal_ci_width" \
    "${wide_ci_args[@]}" \
    --required-suite system_leakage \
    --required-suite public_system_leakage \
    --required-suite public_refusal_safety \
    --required-policy none \
    --required-policy kv_int4_sim \
    --require-causal-patch \
    --require-policy-pinned \
    --required-figure causal_restoration_fraction \
    --required-figure causal_restoration_flow \
    --require-public-provenance
  uv run python scripts/export_human_audit_sample.py \
    --results-dir "$causal_results" \
    --per-suite-policy "$audit_per_suite_policy" \
    --annotator-template-count "$audit_annotator_template_count" \
    --include-hidden-reference
}

write_preliminary_claims() {
  uv run python scripts/assess_claims.py \
    --primary-results-dir "$primary_results" \
    --causal-results-dir "$causal_results" \
    --output-dir paper/generated/preliminary_claim_assessment
  uv run python scripts/plan_registered_followups.py \
    --claim-assessment paper/generated/preliminary_claim_assessment/claim_assessment.json \
    --primary-ci-power "$primary_results/ci_power.json" \
    --causal-ci-power "$causal_results/ci_power.json" \
    --output-dir paper/generated/preliminary_followup_plan
}

write_audit_supported_claims() {
  local primary_audit="paper/audit/h200_qwen_full_sweep_summary/human_audit_summary.json"
  local causal_audit="paper/audit/h200_causal_patch_qwen7b_summary/human_audit_summary.json"
  if [[ ! -f "$primary_audit" || ! -f "$causal_audit" ]]; then
    echo "Audit summaries are not present; skipping audit-supported claim assessment."
    return
  fi
  if ! uv run python scripts/assess_claims.py \
    --primary-results-dir "$primary_results" \
    --causal-results-dir "$causal_results" \
    --primary-audit-summary "$primary_audit" \
    --causal-audit-summary "$causal_audit" \
    --output-dir "$claim_generated" \
    --require-human-audit-support \
    --require-cache-mediated-claim; then
    echo "Positive cache-mediated safety erasure claim did not pass; writing honest non-passing claim assessment."
    uv run python scripts/assess_claims.py \
      --primary-results-dir "$primary_results" \
      --causal-results-dir "$causal_results" \
      --primary-audit-summary "$primary_audit" \
      --causal-audit-summary "$causal_audit" \
      --output-dir "$claim_generated" \
      --require-human-audit-support
  fi
  uv run python scripts/plan_registered_followups.py \
    --claim-assessment "$claim_generated/claim_assessment.json" \
    --primary-ci-power "$primary_results/ci_power.json" \
    --causal-ci-power "$causal_results/ci_power.json" \
    --output-dir paper/generated/registered_followup_plan
}

uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q
rebuild_primary
rebuild_causal
write_preliminary_claims

if [[ "$run_open_judge_audit" == "1" ]]; then
  bash scripts/run_publication_open_judge_audits.sh
  bash scripts/aggregate_publication_human_audits.sh
fi

write_audit_supported_claims

uv run python scripts/post_h200_next_steps.py \
  --output-json paper/generated/post_h200_next_steps.json \
  --output-md paper/generated/post_h200_next_steps.md

if [[ "$attempt_publication_build" == "1" ]]; then
  if ! bash scripts/build_publication_artifacts.sh; then
    echo "Publication build did not pass. See paper/build/publication_status.md and claim assessment artifacts."
    echo "Building an honest evidence-gated PDF/arXiv bundle from the completed artifacts."
    bash scripts/build_evidence_gated_paper_artifacts.sh
  fi
fi

echo "H200 post-causal finalization complete."
