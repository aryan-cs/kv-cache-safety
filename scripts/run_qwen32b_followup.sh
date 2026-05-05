#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$(pwd)/.cache/torch}"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to run Qwen 32B follow-up from a dirty git working tree." >&2
  exit 1
fi

uv sync --frozen --extra dev

public_prompt_limit="${PUBLIC_PROMPT_LIMIT:-650}"
audit_per_suite_policy="${AUDIT_PER_SUITE_POLICY:-10}"
audit_annotator_template_count="${AUDIT_ANNOTATOR_TEMPLATE_COUNT:-2}"
audit_include_hidden_reference="${AUDIT_INCLUDE_HIDDEN_REFERENCE:-1}"
audit_hidden_reference_args=()
if [[ "$audit_include_hidden_reference" == "1" ]]; then
  audit_hidden_reference_args+=(--include-hidden-reference)
fi

uv run python scripts/prepare_data.py --suite all
uv run python scripts/prepare_data.py --source hf --suite cyberec_prompt_injection_leakage --limit "$public_prompt_limit" --output-suite public_system_leakage
uv run python scripts/prepare_data.py --source hf --suite public_refusal_combo --limit "$public_prompt_limit" --output-suite public_refusal_safety
uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit "$public_prompt_limit" --output-suite public_benign_overrefusal
uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit "$public_prompt_limit" --output-suite public_xstest_safe
uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit "$public_prompt_limit" --output-suite public_capability_arc
uv run python scripts/check_prepared_suites.py \
  --min-records 600 \
  --suite-min-records system_leakage=2 \
  --suite-min-records public_xstest_safe=200 \
  --require-public-provenance \
  --suite system_leakage \
  --suite public_system_leakage \
  --suite public_refusal_safety \
  --suite public_benign_overrefusal \
  --suite public_xstest_safe \
  --suite public_capability_arc

uv run python scripts/preflight_h200.py \
  --config configs/experiments/h200_qwen32b_public_followup.yaml

run_id="${QWEN32B_RUN_ID:-h200_qwen32b_public_followup_primary}"
uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_qwen32b_public_followup.yaml \
  --run-id "$run_id" \
  --resume

latest="results/$run_id"
uv run python scripts/aggregate_results.py --results-dir "$latest"
uv run python scripts/make_figures.py --results-dir "$latest"
uv run python scripts/export_paper_assets.py \
  --results-dir "$latest" \
  --paper-dir paper/generated/h200_qwen32b_public_followup \
  --macro-prefix QwenThirtyTwo
uv run python scripts/check_publication_readiness.py \
  --results-dir "$latest" \
  --paper-dir paper/generated/h200_qwen32b_public_followup \
  --min-prompts-per-suite 600 \
  --suite-min-prompts system_leakage=2 \
  --suite-min-prompts public_xstest_safe=200 \
  --max-ci-width 0.10 \
  --required-suite system_leakage \
  --required-suite public_system_leakage \
  --required-suite public_refusal_safety \
  --required-suite public_benign_overrefusal \
  --required-suite public_xstest_safe \
  --required-suite public_capability_arc \
  --required-policy none \
  --required-policy sliding_window \
  --required-policy sink_recent \
  --required-policy kv_int4_sim \
  --require-policy-pinned \
  --required-figure safety_capability_phase_portrait \
  --required-figure selective_safety_erasure_heatmap \
  --required-figure prompt_effect_constellation \
  --required-figure cache_state_fingerprint \
  --required-figure safety_state_atlas \
  --required-figure policy_uncertainty_braid \
  --require-public-provenance
uv run python scripts/export_human_audit_sample.py \
  --results-dir "$latest" \
  --per-suite-policy "$audit_per_suite_policy" \
  --annotator-template-count "$audit_annotator_template_count" \
  "${audit_hidden_reference_args[@]}"

echo "Qwen 32B follow-up complete: $latest"
