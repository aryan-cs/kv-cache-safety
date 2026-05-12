#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

host="${H200_HOST:-uiuc-h200}"
remote_dir="${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}"
stage="${SELECTIVITY_STAGE:-powered}"

if [[ "$stage" != "smoke" && "$stage" != "powered" && "$stage" != "all" ]]; then
  echo "SELECTIVITY_STAGE must be smoke, powered, or all; got ${stage}" >&2
  exit 2
fi

default_models=(
  gpt_oss_20b
  qwen2_5_7b_base
  qwen2_5_7b_instruct
  qwen3_5_9b
  mistral_7b_instruct_v0_3
  olmo3_7b_instruct
  phi4
)

if [[ "${SELECTIVITY_INCLUDE_GATED:-0}" == "1" ]]; then
  default_models+=(llama3_1_8b_instruct gemma2_9b_it)
fi

if [[ -n "${SELECTIVITY_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  model_keys=(${SELECTIVITY_MODELS})
else
  model_keys=("${default_models[@]}")
fi

stages=()
if [[ "$stage" == "all" ]]; then
  stages=(smoke powered)
else
  stages=("$stage")
fi

paths=()
add_if_remote_exists() {
  local path="$1"
  if ssh -n "$host" "test -e '$remote_dir/$path'"; then
    paths+=("$path")
  else
    echo "Skipping absent selectivity artifact: ${host}:${remote_dir}/${path}"
  fi
}

add_if_remote_exists paper/generated/selectivity_panel_phase0_ci_power.json
add_if_remote_exists paper/generated/selectivity_panel_phase0_ci_power.md

for run_stage in "${stages[@]}"; do
  for key in "${model_keys[@]}"; do
    add_if_remote_exists "results/selectivity_h200_${run_stage}_${key}"
    add_if_remote_exists "paper/generated/selectivity_h200_${run_stage}_${key}"
  done
  add_if_remote_exists "results/selectivity_h200_${run_stage}_combined"
  add_if_remote_exists "paper/generated/selectivity_h200_${run_stage}_combined"
done

if [[ "${#paths[@]}" -eq 0 ]]; then
  echo "No selectivity artifacts found on ${host}:${remote_dir}" >&2
  exit 1
fi

H200_HOST="$host" H200_WORKDIR="$remote_dir" bash scripts/fetch_h200_results.sh "${paths[@]}"
