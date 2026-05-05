#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

host="${H200_HOST:-uiuc-h200}"
remote_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
local_log_dir="${LOCAL_H200_LOG_DIR:-logs/h200}"
remote_manifest="${REMOTE_H200_ARTIFACT_MANIFEST:-logs/h200/h200_artifact_manifest_remote.json}"
local_manifest="${LOCAL_H200_ARTIFACT_MANIFEST:-logs/h200/h200_artifact_manifest_local.json}"
compare_report="${H200_ARTIFACT_COMPARE_REPORT:-logs/h200/h200_artifact_manifest_compare.json}"

default_paths=(
  "results/h200_qwen_full_sweep"
  "results/h200_causal_patch_qwen7b"
  "paper/audit/h200_qwen_full_sweep_audit_blinded.csv"
  "paper/audit/h200_qwen_full_sweep_audit_blinded_annotator_01.csv"
  "paper/audit/h200_qwen_full_sweep_audit_blinded_annotator_02.csv"
  "paper/audit/h200_qwen_full_sweep_audit_export_manifest.json"
  "paper/audit/h200_qwen_full_sweep_audit_key.jsonl"
  "paper/audit/h200_causal_patch_qwen7b_audit_blinded.csv"
  "paper/audit/h200_causal_patch_qwen7b_audit_blinded_annotator_01.csv"
  "paper/audit/h200_causal_patch_qwen7b_audit_blinded_annotator_02.csv"
  "paper/audit/h200_causal_patch_qwen7b_audit_export_manifest.json"
  "paper/audit/h200_causal_patch_qwen7b_audit_key.jsonl"
)

remote_generated_paths=(
  "paper/generated/h200_qwen_full_sweep"
  "paper/generated/h200_causal_patch_qwen7b"
  "paper/generated/preliminary_claim_assessment"
  "paper/generated/preliminary_followup_plan"
  "paper/generated/post_h200_next_steps.json"
  "paper/generated/post_h200_next_steps.md"
)

if [[ "${FETCH_H200_REMOTE_GENERATED:-0}" == "1" ]]; then
  default_paths+=("${remote_generated_paths[@]}")
fi

if [[ "$#" -gt 0 ]]; then
  paths=("$@")
else
  paths=("${default_paths[@]}")
fi

safe_artifact_path() {
  local path="$1"
  case "$path" in
    /*|*..*|"") return 1 ;;
    results/*|paper/generated/*|paper/audit/*) return 0 ;;
    *) return 1 ;;
  esac
}

for path in "${paths[@]}"; do
  if ! safe_artifact_path "$path"; then
    echo "Refusing unsafe or unsupported artifact path: $path" >&2
    exit 1
  fi
done

mkdir -p "$local_log_dir"

fetch_with_tar() {
  local path="$1"
  local local_path="$2"
  rm -rf "$local_path"
  mkdir -p "$(dirname "$local_path")"
  ssh -n "$host" "cd '$remote_dir' && tar -cf - '$path'" | tar -xf -
}

remote_manifest_cmd=(uv run python scripts/write_artifact_manifest.py)
local_manifest_cmd=(uv run python scripts/write_artifact_manifest.py)
for path in "${paths[@]}"; do
  remote_manifest_cmd+=(--path "$path")
  local_manifest_cmd+=(--path "$path")
done
remote_manifest_cmd+=(--output "$remote_manifest")
local_manifest_cmd+=(--output "$local_manifest")

printf -v quoted_remote_manifest_cmd "%q " "${remote_manifest_cmd[@]}"
ssh -n "$host" "cd '$remote_dir' && $quoted_remote_manifest_cmd"
scp -q "${host}:${remote_dir}/${remote_manifest}" "${local_log_dir}/$(basename "$remote_manifest")"

for path in "${paths[@]}"; do
  remote_path="${remote_dir}/${path}"
  local_path="$path"
  if ssh -n "$host" "test -d '$remote_path'"; then
    if command -v rsync >/dev/null 2>&1; then
      mkdir -p "$local_path"
      rsync -az --checksum --delete "${host}:${remote_path}/" "${local_path}/"
    else
      fetch_with_tar "$path" "$local_path"
    fi
  elif ssh -n "$host" "test -f '$remote_path'"; then
    if command -v rsync >/dev/null 2>&1; then
      mkdir -p "$(dirname "$local_path")"
      rsync -az --checksum "${host}:${remote_path}" "$local_path"
    else
      fetch_with_tar "$path" "$local_path"
    fi
  else
    echo "Missing requested remote artifact path: $remote_path" >&2
    exit 1
  fi
done

"${local_manifest_cmd[@]}"
uv run python scripts/compare_artifact_manifests.py \
  --expected "${local_log_dir}/$(basename "$remote_manifest")" \
  --actual "$local_manifest" \
  --output-json "$compare_report"

echo "Fetched and verified H200 artifacts:"
printf '  %s\n' "${paths[@]}"
