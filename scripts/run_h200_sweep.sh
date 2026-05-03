#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to run H200 sweep from a dirty git working tree." >&2
  echo "Commit or stash local changes so generated artifacts point to an exact commit." >&2
  exit 1
fi

uv sync --extra dev

uv run python scripts/prepare_data.py --suite all
uv run python scripts/prepare_data.py --source hf --suite advbench --limit 200 --output-suite public_refusal_safety
uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit 200 --output-suite public_benign_overrefusal
uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit 200 --output-suite public_xstest_safe
uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit 200 --output-suite public_capability_arc

uv run python scripts/preflight_h200.py \
  --config configs/experiments/qwen7b_smoke.yaml \
  --config configs/experiments/h200_public_qwen14b.yaml \
  --config configs/experiments/h200_causal_patch_qwen7b.yaml \
  --config configs/experiments/h200_attention_diagnostic_qwen7b.yaml

echo "Running Qwen 7B smoke validation..."
uv run python scripts/run_experiment.py --config configs/experiments/qwen7b_smoke.yaml

latest_smoke="$(ls -td results/qwen7b_smoke_* | head -1)"
uv run python scripts/aggregate_results.py --results-dir "$latest_smoke"
uv run python scripts/make_figures.py --results-dir "$latest_smoke"
uv run python scripts/export_paper_assets.py --results-dir "$latest_smoke" --paper-dir paper/generated/qwen7b_smoke

echo "Running primary H200 Qwen 14B sweep..."
uv run python scripts/run_experiment.py --config configs/experiments/h200_public_qwen14b.yaml

latest_full="$(ls -td results/h200_public_qwen14b_* | head -1)"
uv run python scripts/aggregate_results.py --results-dir "$latest_full"
uv run python scripts/make_figures.py --results-dir "$latest_full"
uv run python scripts/export_paper_assets.py --results-dir "$latest_full" --paper-dir paper/generated/h200_qwen_full_sweep
uv run python scripts/check_publication_readiness.py \
  --results-dir "$latest_full" \
  --min-prompts-per-suite 100 \
  --suite-min-prompts system_leakage=2 \
  --max-ci-width 0.08 \
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
uv run python scripts/export_human_audit_sample.py --results-dir "$latest_full" --per-suite-policy 3

echo "Running causal patch diagnostic on Qwen 7B..."
uv run python scripts/run_experiment.py --config configs/experiments/h200_causal_patch_qwen7b.yaml

latest_causal="$(ls -td results/h200_causal_patch_qwen7b_* | head -1)"
uv run python scripts/aggregate_results.py --results-dir "$latest_causal"
uv run python scripts/make_figures.py --results-dir "$latest_causal"
uv run python scripts/export_paper_assets.py --results-dir "$latest_causal" --paper-dir paper/generated/h200_causal_patch_qwen7b
uv run python scripts/check_publication_readiness.py \
  --results-dir "$latest_causal" \
  --min-prompts-per-suite 100 \
  --suite-min-prompts system_leakage=2 \
  --max-ci-width 0.12 \
  --required-suite system_leakage \
  --required-suite public_refusal_safety \
  --required-policy none \
  --required-policy kv_int4_sim \
  --require-causal-patch \
  --require-policy-pinned \
  --require-public-provenance
uv run python scripts/export_human_audit_sample.py --results-dir "$latest_causal" --per-suite-policy 3

echo "Running attention-policy diagnostic on Qwen 7B..."
uv run python scripts/run_experiment.py --config configs/experiments/h200_attention_diagnostic_qwen7b.yaml

latest_attention="$(ls -td results/h200_attention_diagnostic_qwen7b_* | head -1)"
uv run python scripts/aggregate_results.py --results-dir "$latest_attention"
uv run python scripts/make_figures.py --results-dir "$latest_attention"
uv run python scripts/export_paper_assets.py --results-dir "$latest_attention" --paper-dir paper/generated/h200_attention_diagnostic_qwen7b

echo "Primary sweep complete: $latest_full"
