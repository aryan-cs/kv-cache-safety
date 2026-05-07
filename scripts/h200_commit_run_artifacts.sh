#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <run-dir-relative-to-repo> [branch]" >&2
  exit 2
fi

run_dir="$1"
branch="${2:-master}"

if [[ ! -d "$run_dir" ]]; then
  echo "Run directory not found: $run_dir" >&2
  exit 1
fi

git add -f "$run_dir"

if git diff --cached --quiet -- "$run_dir"; then
  echo "No run artifact changes to commit for $run_dir"
  exit 0
fi

run_id="$(basename "$run_dir")"
git commit -m "Add selectivity run artifacts: ${run_id}"
git push origin "$branch"
