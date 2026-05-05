#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_dir"

host="${H200_HOST:-uiuc-h200}"
remote_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
snapshot_arg="${1:-${SNAPSHOT_DIR:-}}"
resume_config="${RESUME_CONFIG:-configs/experiments/h200_causal_patch_qwen7b.yaml}"

if [[ -z "$snapshot_arg" ]]; then
  echo "Usage: SNAPSHOT_DIR=snapshots/h200/<timestamp> bash scripts/restore_h200_snapshot.sh" >&2
  exit 2
fi

snapshot_dir="$(cd "$(dirname "$snapshot_arg")" && pwd -P)/$(basename "$snapshot_arg")"
if [[ ! -d "$snapshot_dir" ]]; then
  echo "Snapshot directory does not exist: ${snapshot_dir}" >&2
  exit 1
fi

run_id="${RUN_ID:-}"
if [[ -z "$run_id" && -f "$snapshot_dir/snapshot_summary.json" ]]; then
  run_id="$(SNAPSHOT_DIR="$snapshot_dir" UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

summary = json.loads((Path(os.environ["SNAPSHOT_DIR"]) / "snapshot_summary.json").read_text())
print(summary.get("run_id") or "")
PY
)"
fi
if [[ -z "$run_id" ]]; then
  mapfile -t result_dirs < <(find "$snapshot_dir/results" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null || true)
  if [[ "${#result_dirs[@]}" -ne 1 ]]; then
    echo "Could not infer RUN_ID from snapshot; set RUN_ID explicitly." >&2
    exit 2
  fi
  run_id="$(basename "${result_dirs[0]}")"
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid RUN_ID=${run_id}; expected only letters, numbers, dot, underscore, or dash." >&2
  exit 2
fi

snapshot_run_dir="$snapshot_dir/results/$run_id"
required_files=(
  "config.resolved.yaml"
  "environment.json"
  "manifest.json"
  "generations.jsonl"
  "cache_stats.parquet"
)
for file in "${required_files[@]}"; do
  if [[ ! -e "$snapshot_run_dir/$file" ]]; then
    echo "Snapshot is missing required run artifact: results/${run_id}/${file}" >&2
    exit 1
  fi
done

snapshot_manifest="$snapshot_dir/snapshot_manifest.json"
if [[ -f "$snapshot_manifest" ]]; then
  UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/write_artifact_manifest.py \
    --root "$snapshot_dir" \
    --path "results/$run_id" \
    --path logs/h200 \
    --output "$snapshot_dir/snapshot_manifest.local_recheck.json"
  UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/compare_artifact_manifests.py \
    --expected "$snapshot_manifest" \
    --actual "$snapshot_dir/snapshot_manifest.local_recheck.json" \
    --output-json "$snapshot_dir/snapshot_manifest.local_recheck_compare.json"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_path="results/${run_id}.pre_restore.${timestamp}"
ssh -n "$host" "cd '${remote_dir}' && mkdir -p results && if [ -e 'results/${run_id}' ]; then mv 'results/${run_id}' '${backup_path}'; fi"

if command -v rsync >/dev/null 2>&1 && ssh -n "$host" "command -v rsync >/dev/null 2>&1"; then
  ssh -n "$host" "cd '${remote_dir}' && mkdir -p 'results/${run_id}'"
  rsync -az --checksum "$snapshot_run_dir/" "${host}:${remote_dir}/results/${run_id}/"
else
  (cd "$snapshot_dir" && tar -cf - "results/${run_id}") | ssh -n "$host" "cd '${remote_dir}' && tar -xf -"
fi

snapshot_commit=""
if [[ -f "$snapshot_dir/snapshot_summary.json" ]]; then
  snapshot_commit="$(SNAPSHOT_DIR="$snapshot_dir" UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

summary = json.loads((Path(os.environ["SNAPSHOT_DIR"]) / "snapshot_summary.json").read_text())
print(summary.get("remote_git_head") or "")
PY
)"
fi

cat <<EOF
Restored snapshot artifacts to ${host}:${remote_dir}/results/${run_id}
Existing remote artifacts, if any, were moved to ${backup_path}.

Safest resume sequence on the H200:
  ssh ${host}
  cd ${remote_dir}
EOF

if [[ -n "$snapshot_commit" ]]; then
  cat <<EOF
  git fetch origin master
  git checkout ${snapshot_commit}
EOF
else
  cat <<EOF
  git fetch origin master
  git checkout master
  git pull --ff-only origin master
EOF
fi

cat <<EOF
  UV_CACHE_DIR=.cache/uv uv run python scripts/run_experiment.py \\
    --config ${resume_config} \\
    --run-id ${run_id} \\
    --resume

If you intentionally resume from a newer commit after confirming the manifest
matrix is unchanged, prefix the run command with ALLOW_RESUME_GIT_MISMATCH=1.
This script does not start compute or create a launcher process.

Do not resume this restored H200 run on the Mac. If the H200 is still unavailable,
run scripts/run_mac_fallback.sh with its separate mac_* run id instead.
EOF
