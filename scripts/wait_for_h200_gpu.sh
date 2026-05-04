#!/usr/bin/env bash
set -euo pipefail

max_used_mib="${MAX_USED_MIB:-20000}"
max_util_pct="${MAX_UTIL_PCT:-20}"
interval_seconds="${INTERVAL_SECONDS:-300}"
nvidia_smi_timeout_seconds="${NVIDIA_SMI_TIMEOUT_SECONDS:-30}"

nvidia_smi_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${nvidia_smi_timeout_seconds}s" nvidia-smi "$@"
  else
    nvidia-smi "$@"
  fi
}

log_visible_gpu_users() {
  echo "Visible compute apps:"
  nvidia_smi_with_timeout \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null |
    sed 's/^/  /' || echo "  query failed or timed out"
  echo "Process monitor snapshot:"
  nvidia_smi_with_timeout pmon -c 1 2>/dev/null |
    sed 's/^/  /' || echo "  pmon failed or timed out"
}

echo "Waiting for H200 GPU: memory.used <= ${max_used_mib} MiB and utilization <= ${max_util_pct}%"
while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! line="$(nvidia_smi_with_timeout --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)"; then
    echo "$timestamp nvidia-smi gate query failed or timed out after ${nvidia_smi_timeout_seconds}s"
    log_visible_gpu_users
    sleep "$interval_seconds"
    continue
  fi
  used_mib="$(echo "$line" | awk -F, '{gsub(/ /, "", $1); print $1}')"
  util_pct="$(echo "$line" | awk -F, '{gsub(/ /, "", $2); print $2}')"
  if [[ -z "$used_mib" || -z "$util_pct" ]]; then
    echo "$timestamp nvidia-smi gate query returned malformed output: ${line}"
    log_visible_gpu_users
    sleep "$interval_seconds"
    continue
  fi
  echo "$timestamp memory.used=${used_mib}MiB utilization=${util_pct}%"
  if [[ "$used_mib" -le "$max_used_mib" && "$util_pct" -le "$max_util_pct" ]]; then
    echo "GPU appears available."
    exit 0
  fi
  log_visible_gpu_users
  sleep "$interval_seconds"
done
