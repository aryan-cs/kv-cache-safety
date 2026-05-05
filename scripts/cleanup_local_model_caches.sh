#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_dir"

dry_run=1
allow_global=0
for arg in "$@"; do
  case "$arg" in
    --yes)
      dry_run=0
      ;;
    --allow-global)
      allow_global=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/cleanup_local_model_caches.sh [--yes] [--allow-global]

Dry-runs by default. With --yes, deletes only model/cache directories that
resolve inside this repository unless --allow-global is also passed.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

candidate_paths=(
  "$repo_dir/.cache/mac_fallback/huggingface"
  "$repo_dir/.cache/mac_fallback/torch"
  "$repo_dir/.cache/huggingface"
  "$repo_dir/.cache/torch"
  "$repo_dir/hf_cache"
  "$repo_dir/models"
)

for var in HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME; do
  value="${!var:-}"
  if [[ -n "$value" ]]; then
    candidate_paths+=("$value")
  fi
done

seen=""
for raw_path in "${candidate_paths[@]}"; do
  [[ -z "$raw_path" ]] && continue
  [[ -e "$raw_path" ]] || continue
  parent="$(cd "$(dirname "$raw_path")" && pwd -P)"
  resolved="${parent}/$(basename "$raw_path")"
  case "$seen" in
    *"|$resolved|"*) continue ;;
  esac
  seen="${seen}|${resolved}|"

  case "$resolved" in
    "$repo_dir/.cache/uv"|"$repo_dir/.cache/uv/"*|"$repo_dir/results"|"$repo_dir/results/"*|"$repo_dir/snapshots"|"$repo_dir/snapshots/"*)
      echo "Skipping protected path: $resolved"
      continue
      ;;
  esac

  case "$resolved" in
    "$repo_dir"/*)
      ;;
    *)
      if [[ "$allow_global" != "1" ]]; then
        echo "Skipping non-repo cache path without --allow-global: $resolved"
        continue
      fi
      ;;
  esac

  if [[ "$dry_run" == "1" ]]; then
    echo "Would delete: $resolved"
  else
    rm -rf "$resolved"
    echo "Deleted: $resolved"
  fi
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run only. Re-run with --yes to delete the listed repo-local model caches."
fi
