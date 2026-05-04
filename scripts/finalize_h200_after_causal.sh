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
primary_generated="${PRIMARY_GENERATED_DIR:-paper/generated/$(basename "$primary_results")}"
causal_generated="${CAUSAL_GENERATED_DIR:-paper/generated/$(basename "$causal_results")}"
claim_generated="${CLAIM_GENERATED_DIR:-paper/generated/claim_assessment}"
merged_primary_run_id="${MERGED_PRIMARY_RUN_ID:-h200_qwen_full_sweep_plus_ci_extension}"
merged_primary_results="${MERGED_PRIMARY_RESULTS_DIR:-results/$merged_primary_run_id}"
merged_primary_generated="${MERGED_PRIMARY_GENERATED_DIR:-paper/generated/$merged_primary_run_id}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
causal_ci_width="${CAUSAL_CI_WIDTH:-0.12}"
audit_per_suite_policy="${AUDIT_PER_SUITE_POLICY:-10}"
audit_annotator_template_count="${AUDIT_ANNOTATOR_TEMPLATE_COUNT:-2}"
run_open_judge_audit="${RUN_OPEN_JUDGE_AUDIT:-0}"
attempt_publication_build="${ATTEMPT_PUBLICATION_BUILD:-1}"
staging_allow_wide_ci="${FINALIZER_ALLOW_WIDE_CI:-0}"
allow_evidence_gated_fallback="${ALLOW_EVIDENCE_GATED_FALLBACK:-0}"
run_ci_extension_if_needed="${RUN_CI_EXTENSION_IF_NEEDED:-1}"
primary_audit_summary_override="${PRIMARY_AUDIT_SUMMARY_DIR:-}"
causal_audit_summary_override="${CAUSAL_AUDIT_SUMMARY_DIR:-}"

update_run_context() {
  primary_run_id="$(basename "$primary_results")"
  causal_run_id="$(basename "$causal_results")"
  if [[ -n "$primary_audit_summary_override" ]]; then
    primary_audit_summary="$primary_audit_summary_override"
  else
    primary_audit_summary="paper/audit/${primary_run_id}_summary"
  fi
  if [[ -n "$causal_audit_summary_override" ]]; then
    causal_audit_summary="$causal_audit_summary_override"
  else
    causal_audit_summary="paper/audit/${causal_run_id}_summary"
  fi
}

use_merged_primary_evidence() {
  primary_results="$merged_primary_results"
  primary_generated="$merged_primary_generated"
  update_run_context
}

update_run_context

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

check_primary_readiness() {
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$primary_results" \
    --paper-dir "$primary_generated" \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --suite-min-prompts public_xstest_safe=200 \
    --max-ci-width "$target_ci_width" \
    "${wide_ci_args[@]}" \
    "$@" \
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
}

maybe_run_ci_extension_for_primary() {
  if [[ "$run_ci_extension_if_needed" != "1" ]]; then
    return 1
  fi
  if [[ "$primary_results" == "$merged_primary_results" ]]; then
    return 1
  fi
  if ! check_primary_readiness --allow-wide-ci; then
    return 1
  fi
  echo "Primary run passes non-CI readiness but misses the strict CI-width gate."
  echo "Running registered CI extension and promoting merged primary evidence."
  PRIMARY_RESULTS_DIR="$primary_results" \
  MERGED_PRIMARY_RUN_ID="$merged_primary_run_id" \
  TARGET_CI_WIDTH="$target_ci_width" \
  AUDIT_ANNOTATOR_TEMPLATE_COUNT="$audit_annotator_template_count" \
  bash scripts/run_h200_ci_extension.sh
  use_merged_primary_evidence
  return 0
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
  if ! check_primary_readiness; then
    if maybe_run_ci_extension_for_primary; then
      rebuild_primary
      return
    fi
    echo "Primary readiness failed and was not resolved by the CI-extension gate." >&2
    exit 1
  fi
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
  local primary_audit="$primary_audit_summary/human_audit_summary.json"
  local causal_audit="$causal_audit_summary/human_audit_summary.json"
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

run_publication_artifact_build() {
  PRIMARY_RESULTS_DIR="$primary_results" \
  CAUSAL_RESULTS_DIR="$causal_results" \
  PRIMARY_GENERATED_DIR="$primary_generated" \
  CAUSAL_GENERATED_DIR="$causal_generated" \
  PRIMARY_AUDIT_SUMMARY_DIR="$primary_audit_summary" \
  CAUSAL_AUDIT_SUMMARY_DIR="$causal_audit_summary" \
  CLAIM_GENERATED_DIR="$claim_generated" \
  bash scripts/build_publication_artifacts.sh
}

run_evidence_gated_artifact_build() {
  PRIMARY_RESULTS_DIR="$primary_results" \
  CAUSAL_RESULTS_DIR="$causal_results" \
  PRIMARY_GENERATED_DIR="$primary_generated" \
  CAUSAL_GENERATED_DIR="$causal_generated" \
  PRIMARY_AUDIT_SUMMARY_DIR="$primary_audit_summary" \
  CAUSAL_AUDIT_SUMMARY_DIR="$causal_audit_summary" \
  CLAIM_GENERATED_DIR="$claim_generated" \
  bash scripts/build_evidence_gated_paper_artifacts.sh
}

uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q
rebuild_primary
rebuild_causal
write_preliminary_claims

if [[ "$run_open_judge_audit" == "1" ]]; then
  PRIMARY_RUN_ID="$primary_run_id" \
  CAUSAL_RUN_ID="$causal_run_id" \
  bash scripts/run_publication_open_judge_audits.sh
  PRIMARY_RUN_ID="$primary_run_id" \
  CAUSAL_RUN_ID="$causal_run_id" \
  PRIMARY_RESULTS_DIR="$primary_results" \
  CAUSAL_RESULTS_DIR="$causal_results" \
  PRIMARY_AUDIT_SUMMARY_DIR="$primary_audit_summary" \
  CAUSAL_AUDIT_SUMMARY_DIR="$causal_audit_summary" \
  AUDIT_SOURCE=open_judge \
  bash scripts/aggregate_publication_human_audits.sh
fi

write_audit_supported_claims

uv run python scripts/post_h200_next_steps.py \
  --primary-results-dir "$primary_results" \
  --causal-results-dir "$causal_results" \
  --primary-generated-dir "$primary_generated" \
  --causal-generated-dir "$causal_generated" \
  --primary-audit-dir "$primary_audit_summary" \
  --causal-audit-dir "$causal_audit_summary" \
  --claim-assessment "$claim_generated/claim_assessment.json" \
  --arxiv-source-dir paper/build/arxiv_source \
  --arxiv-archive paper/build/arxiv_source.tar.gz \
  --output-json paper/generated/post_h200_next_steps.json \
  --output-md paper/generated/post_h200_next_steps.md

if [[ "$attempt_publication_build" == "1" ]]; then
  if ! run_publication_artifact_build; then
    echo "Publication build did not pass. See paper/build/publication_status.md and claim assessment artifacts."
    echo "Building an honest evidence-gated PDF/arXiv bundle from the completed artifacts."
    run_evidence_gated_artifact_build
    if [[ "$allow_evidence_gated_fallback" != "1" ]]; then
      echo "Evidence-gated fallback was built, but strict publication readiness did not pass." >&2
      echo "Set ALLOW_EVIDENCE_GATED_FALLBACK=1 only when intentionally producing a non-publication fallback bundle." >&2
      exit 1
    fi
  fi
fi

echo "H200 post-causal finalization complete."
