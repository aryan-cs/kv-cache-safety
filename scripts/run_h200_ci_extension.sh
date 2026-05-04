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
  echo "Refusing to run H200 CI extension from a dirty git working tree." >&2
  echo "Commit or stash local changes so generated artifacts point to an exact commit." >&2
  exit 1
fi

uv sync --frozen --extra dev

ci_prompt_limit="${CI_PROMPT_LIMIT:-650}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"

uv run python scripts/prepare_data.py --suite all
uv run python scripts/prepare_data.py --source hf --suite cyberec_prompt_injection_leakage --limit "$ci_prompt_limit" --output-suite public_system_leakage
uv run python scripts/prepare_data.py --source hf --suite public_refusal_combo --limit "$ci_prompt_limit" --output-suite public_refusal_safety
uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit "$ci_prompt_limit" --output-suite public_benign_overrefusal
uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit "$ci_prompt_limit" --output-suite public_xstest_safe
uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit "$ci_prompt_limit" --output-suite public_capability_arc

uv run python scripts/preflight_h200.py \
  --config configs/experiments/h200_qwen14b_ci_extension.yaml

run_id="${CI_EXTENSION_RUN_ID:-h200_qwen14b_ci_extension_primary}"
uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_qwen14b_ci_extension.yaml \
  --run-id "$run_id" \
  --resume

latest="results/$run_id"
paper_dir="paper/generated/h200_qwen14b_ci_extension"
uv run python scripts/aggregate_results.py --results-dir "$latest"
uv run python scripts/make_figures.py --results-dir "$latest"
uv run python scripts/export_paper_assets.py \
  --results-dir "$latest" \
  --paper-dir "$paper_dir" \
  --macro-prefix CIExtension
uv run python scripts/plan_ci_power.py \
  --results-dir "$latest" \
  --target-ci-width "$target_ci_width" \
  --output-json "$latest/ci_power.json" \
  --output-md "$paper_dir/ci_power.md"
uv run python scripts/check_publication_readiness.py \
  --results-dir "$latest" \
  --paper-dir "$paper_dir" \
  --min-prompts-per-suite 600 \
  --suite-min-prompts system_leakage=2 \
  --suite-min-prompts public_xstest_safe=200 \
  --max-ci-width "$target_ci_width" \
  --required-suite system_leakage \
  --required-suite public_system_leakage \
  --required-suite public_refusal_safety \
  --required-suite public_benign_overrefusal \
  --required-suite public_xstest_safe \
  --required-suite public_capability_arc \
  --required-policy none \
  --required-policy sliding_window \
  --required-policy kv_int4_sim \
  --require-policy-pinned \
  --require-public-provenance
uv run python scripts/export_human_audit_sample.py --results-dir "$latest" --per-suite-policy 10

echo "CI-extension sweep complete: $latest"
