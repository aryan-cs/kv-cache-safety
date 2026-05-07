#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <remote-run-dir-relative-to-repo> [local-results-dir]" >&2
  exit 2
fi

remote_run_dir="$1"
local_results_dir="${2:-results}"
remote_root="/home/aryang9/sandbox/llm-safety"
remote="uiuc-h200"

mkdir -p "$local_results_dir"

rsync -az --partial --info=progress2 \
  "${remote}:${remote_root}/${remote_run_dir%/}/" \
  "${local_results_dir}/$(basename "$remote_run_dir")/"

echo "Fetched ${remote}:${remote_root}/${remote_run_dir} -> ${local_results_dir}/$(basename "$remote_run_dir")"
