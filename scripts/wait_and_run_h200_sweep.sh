#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
expected_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
branch="${BRANCH:-master}"
sweep_script="${SWEEP_SCRIPT:-scripts/run_h200_sweep.sh}"

if [[ "$repo_dir" != "$expected_dir" ]]; then
  echo "Refusing to run outside ${expected_dir}: ${repo_dir}" >&2
  exit 1
fi

case "$sweep_script" in
  scripts/run_h200_sweep.sh|scripts/run_h200_ci_extension.sh|scripts/run_qwen32b_followup.sh) ;;
  *)
    echo "Refusing unrecognized sweep script: ${sweep_script}" >&2
    exit 1
    ;;
esac

cd "$repo_dir"
mkdir -p .locks logs/h200

lock_dir=".locks/h200_sweep.lock"
if mkdir "$lock_dir" 2>/dev/null; then
  echo "$$" > "${lock_dir}/pid"
else
  existing_pid="$(cat "${lock_dir}/pid" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Another H200 sweep launcher is already running with pid ${existing_pid}." >&2
    exit 1
  fi
  rm -f "${lock_dir}/pid"
  rmdir "$lock_dir" 2>/dev/null || {
    echo "Found a stale non-empty lock at ${lock_dir}; inspect it before retrying." >&2
    exit 1
  }
  mkdir "$lock_dir"
  echo "$$" > "${lock_dir}/pid"
fi
cleanup() {
  rm -f "${lock_dir}/pid"
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="logs/h200/wait_and_run_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "H200 sweep launcher started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repository: ${repo_dir}"
echo "Branch: ${branch}"
echo "Sweep script: ${sweep_script}"
echo "Log: ${log_file}"

sync_and_validate() {
  local phase="$1"
  echo "${phase}: syncing ${branch} and running static validation..."
  git fetch origin "$branch"
  git checkout "$branch"
  git pull --ff-only origin "$branch"

  if [[ -n "$(git status --short)" ]]; then
    echo "Refusing to run from a dirty H200 worktree." >&2
    git status --short >&2
    exit 1
  fi

  uv run ruff check .
  uv run pytest -q
  bash -n scripts/run_h200_sweep.sh scripts/run_h200_ci_extension.sh scripts/run_qwen32b_followup.sh
}

sync_and_validate "Pre-gate"

bash scripts/wait_for_h200_gpu.sh

sync_and_validate "Post-gate"

echo "GPU gate passed at $(date -u +%Y-%m-%dT%H:%M:%SZ); starting ${sweep_script}."
bash "$sweep_script"
echo "H200 sweep launcher finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
