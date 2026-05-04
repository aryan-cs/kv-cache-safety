#!/usr/bin/env bash
set -euo pipefail

max_used_mib="${MAX_USED_MIB:-20000}"
max_util_pct="${MAX_UTIL_PCT:-20}"
interval_seconds="${INTERVAL_SECONDS:-300}"

log_visible_gpu_users() {
  echo "Visible compute apps:"
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null |
    sed 's/^/  /' || true
  echo "Process monitor snapshot:"
  nvidia-smi pmon -c 1 2>/dev/null | sed 's/^/  /' || true
}

echo "Waiting for H200 GPU: memory.used <= ${max_used_mib} MiB and utilization <= ${max_util_pct}%"
while true; do
  line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)"
  used_mib="$(echo "$line" | awk -F, '{gsub(/ /, "", $1); print $1}')"
  util_pct="$(echo "$line" | awk -F, '{gsub(/ /, "", $2); print $2}')"
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$timestamp memory.used=${used_mib}MiB utilization=${util_pct}%"
  if [[ "$used_mib" -le "$max_used_mib" && "$util_pct" -le "$max_util_pct" ]]; then
    echo "GPU appears available."
    exit 0
  fi
  log_visible_gpu_users
  sleep "$interval_seconds"
done
