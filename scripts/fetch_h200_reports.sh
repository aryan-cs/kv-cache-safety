#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

host="${H200_HOST:-uiuc-h200}"
remote_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
local_dir="${LOCAL_H200_LOG_DIR:-logs/h200}"

mkdir -p "$local_dir"

for name in h200_admin_report.md h200_status_latest.md h200_status_latest.json; do
  scp -q "${host}:${remote_dir}/logs/h200/${name}" "${local_dir}/${name}"
done

echo "Fetched H200 reports into ${local_dir}:"
ls -lh \
  "${local_dir}/h200_admin_report.md" \
  "${local_dir}/h200_status_latest.md" \
  "${local_dir}/h200_status_latest.json"
