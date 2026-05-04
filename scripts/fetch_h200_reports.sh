#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

host="${H200_HOST:-uiuc-h200}"
remote_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
local_dir="${LOCAL_H200_LOG_DIR:-logs/h200}"

mkdir -p "$local_dir"

fetch_remote_report() {
  local remote_name="$1"
  local local_name="$2"
  local remote_path="${remote_dir}/logs/h200/${remote_name}"
  if ssh -q "$host" "test -f '${remote_path}'"; then
    scp -q "${host}:${remote_path}" "${local_dir}/${local_name}"
    return 0
  fi
  return 1
}

if ! fetch_remote_report h200_admin_report_latest.md h200_admin_report.md; then
  fetch_remote_report h200_admin_report.md h200_admin_report.md
fi
fetch_remote_report h200_status_latest.md h200_status_latest.md
fetch_remote_report h200_status_latest.json h200_status_latest.json

echo "Fetched H200 reports into ${local_dir}:"
ls -lh \
  "${local_dir}/h200_admin_report.md" \
  "${local_dir}/h200_status_latest.md" \
  "${local_dir}/h200_status_latest.json"
