#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

uv sync --extra dev
uv run python scripts/run_experiment.py --config configs/experiments/qwen32b_followup.yaml

latest="$(ls -td results/qwen32b_followup_* | head -1)"
uv run python scripts/aggregate_results.py --results-dir "$latest"
uv run python scripts/make_figures.py --results-dir "$latest"
uv run python scripts/export_paper_assets.py --results-dir "$latest" --paper-dir paper/generated/qwen32b_followup

echo "Qwen 32B follow-up complete: $latest"
