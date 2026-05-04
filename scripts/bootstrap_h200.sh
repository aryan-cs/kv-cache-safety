#!/usr/bin/env bash
set -euo pipefail

workdir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
repo_url="${REPO_URL:-https://github.com/aryan-cs/llm-safety.git}"
branch="${BRANCH:-master}"

if [[ "$workdir" != "/home/aryang9/sandbox/llm-safety" ]]; then
  echo "Refusing to bootstrap outside /home/aryang9/sandbox/llm-safety: $workdir" >&2
  exit 1
fi

if [[ ! -d "$(dirname "$workdir")" || ! -w "$(dirname "$workdir")" ]]; then
  echo "Parent directory is missing or not writable: $(dirname "$workdir")" >&2
  exit 1
fi

if [[ -d "$workdir/.git" ]]; then
  cd "$workdir"
  git config --global --add safe.directory "$workdir" || true
  git fetch origin "$branch"
  git checkout "$branch"
  git pull --ff-only origin "$branch"
elif [[ -z "$(find "$workdir" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
  rm -rf "$workdir"
  git clone --branch "$branch" "$repo_url" "$workdir"
  cd "$workdir"
  git config --global --add safe.directory "$workdir" || true
else
  echo "Refusing to clone into non-empty non-git directory: $workdir" >&2
  exit 1
fi

if [[ "$(pwd -P)" != "$workdir" ]]; then
  echo "Unexpected working directory after bootstrap: $(pwd -P)" >&2
  exit 1
fi

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to validate a dirty H200 worktree." >&2
  git status --short >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$workdir/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$workdir/.cache/torch}"

uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q

uv run python scripts/prepare_data.py --suite all
uv run python scripts/prepare_data.py --source hf --suite public_refusal_combo --limit 200 --output-suite public_refusal_safety
uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit 200 --output-suite public_benign_overrefusal
uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit 200 --output-suite public_xstest_safe
uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit 200 --output-suite public_capability_arc

preflight_args=(
  --config configs/experiments/qwen7b_smoke.yaml
  --config configs/experiments/h200_public_qwen14b.yaml
  --config configs/experiments/h200_causal_patch_qwen7b.yaml
  --config configs/experiments/h200_attention_diagnostic_qwen7b.yaml
)
if [[ "${SKIP_MODEL_CONFIG_CHECK:-0}" == "1" ]]; then
  preflight_args+=(--skip-model-config-check)
fi

uv run python scripts/preflight_h200.py "${preflight_args[@]}"

echo "H200 bootstrap complete at $(pwd -P)"
