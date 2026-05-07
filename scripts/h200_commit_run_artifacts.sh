#!/usr/bin/env bash
set -euo pipefail

branch="${BRANCH:-master}"
max_git_artifact_bytes="${MAX_GIT_ARTIFACT_BYTES:-95000000}"
paths=()

usage() {
  echo "Usage: $0 [--branch <branch>] <artifact-path> [<artifact-path> ...]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

if [[ $# -eq 2 && "$1" != "--branch" && "$2" != "--branch" && -e "$1" && ! -e "$2" ]]; then
  paths+=("$1")
  branch="$2"
  set --
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      branch="$2"
      shift 2
      ;;
    *)
      paths+=("$1")
      shift
      ;;
  esac
done

if [[ "${#paths[@]}" -eq 0 ]]; then
  usage
  exit 2
fi

for path in "${paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Artifact path not found: $path" >&2
    exit 1
  fi
done

stage_artifact_path() {
  local path="$1"
  local bytes

  if [[ -f "$path" ]]; then
    bytes="$(wc -c < "$path" | tr -d '[:space:]')"
    if [[ "$bytes" -gt "$max_git_artifact_bytes" ]]; then
      echo "Skipping oversized artifact for git (${bytes} bytes): $path" >&2
      return 0
    fi
    git add -f -- "$path"
    return 0
  fi

  if [[ -d "$path" ]]; then
    while IFS= read -r -d '' file; do
      stage_artifact_path "$file"
    done < <(find "$path" -type f -print0)
    return 0
  fi

  git add -f -- "$path"
}

for path in "${paths[@]}"; do
  stage_artifact_path "$path"
done

if git diff --cached --quiet -- "${paths[@]}"; then
  echo "No artifact changes to commit for: ${paths[*]}"
  exit 0
fi

if [[ "${#paths[@]}" -eq 1 ]]; then
  artifact_label="$(basename "${paths[0]}")"
else
  artifact_label="$(basename "${paths[0]}") plus related artifacts"
fi

git commit -m "Add H200 artifacts: ${artifact_label}" -- "${paths[@]}"
git push origin HEAD:"$branch"
