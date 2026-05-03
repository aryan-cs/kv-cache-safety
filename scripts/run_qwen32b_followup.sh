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

uv run python scripts/prepare_data.py --suite all
uv run python scripts/prepare_data.py --source hf --suite advbench --limit "$public_prompt_limit" --output-suite public_refusal_safety
uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit "$public_prompt_limit" --output-suite public_benign_overrefusal
uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit "$public_prompt_limit" --output-suite public_xstest_safe
uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit "$public_prompt_limit" --output-suite public_capability_arc

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
uv run python scripts/export_human_audit_sample.py \
  --results-dir "$latest" \
  --per-suite-policy "$audit_per_suite_policy"

echo "Qwen 32B follow-up complete: $latest"
