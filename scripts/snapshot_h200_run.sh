#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_dir"

host="${H200_HOST:-uiuc-h200}"
remote_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
run_id="${RUN_ID:-h200_causal_patch_qwen7b}"
snapshot_root="${SNAPSHOT_ROOT:-snapshots/h200}"
timestamp="${SNAPSHOT_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
log_tail_bytes="${H200_SNAPSHOT_LOG_TAIL_BYTES:-262144}"
log_file_limit="${H200_SNAPSHOT_LOG_FILE_LIMIT:-0}"
log_tail_timeout_seconds="${H200_SNAPSHOT_LOG_TIMEOUT_SECONDS:-15}"

if [[ ! "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid RUN_ID=${run_id}; expected only letters, numbers, dot, underscore, or dash." >&2
  exit 2
fi
if [[ ! "$log_tail_bytes" =~ ^[0-9]+$ ]]; then
  echo "Invalid H200_SNAPSHOT_LOG_TAIL_BYTES=${log_tail_bytes}; expected an integer." >&2
  exit 2
fi
if [[ ! "$log_file_limit" =~ ^[0-9]+$ ]]; then
  echo "Invalid H200_SNAPSHOT_LOG_FILE_LIMIT=${log_file_limit}; expected an integer." >&2
  exit 2
fi
if [[ ! "$log_tail_timeout_seconds" =~ ^[0-9]+$ ]]; then
  echo "Invalid H200_SNAPSHOT_LOG_TIMEOUT_SECONDS=${log_tail_timeout_seconds}; expected an integer." >&2
  exit 2
fi

snapshot_dir="${snapshot_root%/}/${timestamp}"
mkdir -p "$snapshot_dir"

remote_path_exists() {
  local rel="$1"
  ssh -n "$host" "test -e '${remote_dir}/${rel}'"
}

copy_remote_path() {
  local rel="$1"
  local dest="$snapshot_dir/$rel"
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  if command -v rsync >/dev/null 2>&1 && ssh -n "$host" "command -v rsync >/dev/null 2>&1"; then
    if ssh -n "$host" "test -d '${remote_dir}/${rel}'"; then
      mkdir -p "$dest"
      rsync -az --checksum "${host}:${remote_dir}/${rel}/" "$dest/"
    else
      rsync -az --checksum "${host}:${remote_dir}/${rel}" "$dest"
    fi
  else
    ssh -n "$host" \
      "cd '${remote_dir}' && tar --warning=no-file-changed --ignore-failed-read -cf - '${rel}'" \
      | tar -xf - -C "$snapshot_dir"
  fi
}

copy_remote_log_file() {
  local rel="$1"
  local dest="$snapshot_dir/$rel"
  rm -f "$dest"
  mkdir -p "$(dirname "$dest")"
  if ! ssh -n "$host" \
    "timeout '${log_tail_timeout_seconds}' tail -c '${log_tail_bytes}' '${remote_dir}/${rel}'" \
    > "$dest"; then
    printf 'Log snapshot timed out or failed for %s at %s\n' \
      "$rel" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dest"
  fi
}

required_result_path="results/${run_id}"
if ! remote_path_exists "$required_result_path"; then
  echo "Remote run directory does not exist: ${host}:${remote_dir}/${required_result_path}" >&2
  exit 1
fi

copy_remote_path "$required_result_path"

log_list="$snapshot_dir/remote_log_paths.txt"
: > "$log_list"
if ssh -n "$host" "cd '${remote_dir}' && test -d logs/h200"; then
  ssh -n "$host" \
    "cd '${remote_dir}' && { ls -1t logs/h200/wait_and_run_*.log logs/h200/finalize_after_causal_watch_*.log logs/h200/finalize_after_causal_watch_launcher.out logs/h200/launcher.out 2>/dev/null || true; } | head -n '${log_file_limit}'" \
    > "$log_list"
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    copy_remote_log_file "$rel"
  done < "$log_list"
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/write_artifact_manifest.py \
  --root "$snapshot_dir" \
  --path "$required_result_path" \
  --path logs/h200 \
  --output "$snapshot_dir/snapshot_manifest.json"

SNAPSHOT_DIR="$snapshot_dir" \
RUN_ID="$run_id" \
H200_HOST_VALUE="$host" \
H200_WORKDIR_VALUE="$remote_dir" \
UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python - <<'PY'
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

snapshot_dir = Path(os.environ["SNAPSHOT_DIR"])
run_id = os.environ["RUN_ID"]
run_dir = snapshot_dir / "results" / run_id
generations = run_dir / "generations.jsonl"
manifest = run_dir / "manifest.json"
cache_stats = run_dir / "cache_stats.parquet"

generation_rows = 0
if generations.exists():
    with generations.open("r", encoding="utf-8") as f:
        generation_rows = sum(1 for line in f if line.strip())

expected_generation_count = None
run_manifest = {}
if manifest.exists():
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    expected_generation_count = run_manifest.get("expected_generation_count")

summary = {
    "schema_version": 1,
    "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "snapshot_dir": str(snapshot_dir),
    "run_id": run_id,
    "remote_host": os.environ["H200_HOST_VALUE"],
    "remote_workdir": os.environ["H200_WORKDIR_VALUE"],
    "remote_git_head": run_manifest.get("git_commit"),
    "remote_origin_master": None,
    "remote_git_status_short": None,
    "generation_rows": generation_rows,
    "expected_generation_count": expected_generation_count,
    "metrics_present": (run_dir / "metrics.json").exists(),
    "cache_stats_present": cache_stats.exists(),
    "cache_stats_bytes": cache_stats.stat().st_size if cache_stats.exists() else 0,
}
output = snapshot_dir / "snapshot_summary.json"
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

ln -sfn "$timestamp" "${snapshot_root%/}/latest"

cat <<EOF

Snapshot complete: ${snapshot_dir}

To restore this run onto a restarted H200 checkout:
  SNAPSHOT_DIR=${snapshot_dir} bash scripts/restore_h200_snapshot.sh

The restore script only copies artifacts back. It prints the explicit resume
command and does not start a duplicate experiment process.
EOF
