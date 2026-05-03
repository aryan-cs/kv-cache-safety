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
  git fetch origin "$branch"
  git checkout "$branch"
  git pull --ff-only origin "$branch"
elif [[ -z "$(find "$workdir" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
  rm -rf "$workdir"
  git clone --branch "$branch" "$repo_url" "$workdir"
  cd "$workdir"
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

uv sync --extra dev
uv run ruff check .
uv run pytest -q

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
