#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_HOME="${TORCH_HOME:-$(pwd)/.cache/torch}"

if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
fi

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to run selectivity panel from a dirty git working tree." >&2
  echo "Commit or stash local changes so generated artifacts point to an exact commit." >&2
  exit 1
fi

stage="${SELECTIVITY_STAGE:-smoke}"
if [[ "$stage" != "smoke" && "$stage" != "powered" && "$stage" != "all" ]]; then
  echo "SELECTIVITY_STAGE must be smoke, powered, or all; got ${stage}" >&2
  exit 2
fi

public_prompt_limit="${PUBLIC_PROMPT_LIMIT:-1300}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
commit_run_artifacts="${COMMIT_RUN_ARTIFACTS:-0}"
branch="${BRANCH:-master}"

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

uv sync --frozen --extra dev
uv run python scripts/generate_selectivity_configs.py --stage smoke --overwrite
uv run python scripts/generate_selectivity_configs.py --stage powered --overwrite
uv run python scripts/prepare_data.py --suite all

register_power_plan() {
  uv run python scripts/plan_ci_power.py \
    --target-ci-width "$target_ci_width" \
    --output-json docs/generated/selectivity_panel_phase0_ci_power.json \
    --output-md docs/generated/selectivity_panel_phase0_ci_power.md
}

prepare_powered_data() {
  uv run python scripts/prepare_data.py --source hf --suite cyberec_prompt_injection_leakage --limit "$public_prompt_limit" --output-suite public_system_leakage
  uv run python scripts/prepare_data.py --source hf --suite public_refusal_combo --limit "$public_prompt_limit" --output-suite public_refusal_safety
  uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit "$public_prompt_limit" --output-suite public_benign_overrefusal
  uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit "$public_prompt_limit" --output-suite public_xstest_safe
  uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit "$public_prompt_limit" --output-suite public_capability_arc
  uv run python scripts/check_prepared_suites.py \
    --min-records 1200 \
    --suite-min-records system_leakage=2 \
    --suite-min-records adversarial_refusal_safety=3 \
    --suite-min-records public_xstest_safe=1200 \
    --require-public-provenance \
    --suite public_system_leakage \
    --suite public_refusal_safety \
    --suite public_benign_overrefusal \
    --suite public_xstest_safe \
    --suite public_capability_arc
}

config_has_null_revision() {
  local config="$1"
  grep -Eq '^[[:space:]]*revision:[[:space:]]*(null)?[[:space:]]*$' "$config"
}

run_config() {
  local run_stage="$1"
  local key="$2"
  local config="configs/experiments/selectivity_h200_${run_stage}_${key}.yaml"
  local run_id="selectivity_h200_${run_stage}_${key}"
  local results_dir="results/${run_id}"
  local generated_dir="docs/generated/${run_id}"

  if [[ ! -f "$config" ]]; then
    echo "Skipping missing selectivity config: ${config}" >&2
    return 0
  fi
  if config_has_null_revision "$config"; then
    echo "Skipping ${config}: model revision is not pinned or access is unavailable." >&2
    return 0
  fi

  echo "Preflighting ${config}"
  uv run python scripts/preflight_h200.py --config "$config"

  echo "Running ${run_id}"
  uv run python scripts/run_experiment.py --config "$config" --run-id "$run_id" --resume
  if grep -q 'track: base_model' "$config"; then
    uv run python scripts/score_base_model_track.py --config "$config" --run-id "$run_id" --resume
  fi
  uv run python scripts/aggregate_results.py --results-dir "$results_dir"
  uv run python scripts/make_figures.py --results-dir "$results_dir"
  uv run python scripts/export_paper_assets.py --results-dir "$results_dir" --paper-dir "$generated_dir"
  uv run python scripts/report_selectivity_status.py --run-dir "$results_dir"

  if [[ "$run_stage" == "powered" ]]; then
    if grep -q 'track: base_model' "$config"; then
      uv run python scripts/check_publication_readiness.py \
        --results-dir "$results_dir" \
        --paper-dir "$generated_dir" \
        --min-prompts-per-suite 1200 \
        --suite-min-prompts instruction_following=2 \
        --max-ci-width "$target_ci_width" \
        --required-policy none \
        --required-policy sliding_window \
        --required-policy sink_recent \
        --require-public-provenance
    else
      uv run python scripts/check_publication_readiness.py \
        --results-dir "$results_dir" \
        --paper-dir "$generated_dir" \
        --min-prompts-per-suite 1200 \
        --suite-min-prompts system_leakage=2 \
        --suite-min-prompts adversarial_refusal_safety=3 \
        --suite-min-prompts public_xstest_safe=1200 \
        --max-ci-width "$target_ci_width" \
        --ci-width-exempt-suite system_leakage \
        --ci-width-exempt-suite adversarial_refusal_safety \
        --required-suite system_leakage \
        --required-suite public_system_leakage \
        --required-suite public_refusal_safety \
        --required-suite adversarial_refusal_safety \
        --required-suite public_benign_overrefusal \
        --required-suite public_xstest_safe \
        --required-suite public_capability_arc \
        --required-policy none \
        --required-policy sliding_window \
        --required-policy sink_recent \
        --required-policy random_matched \
        --require-policy-pinned \
        --require-public-provenance
    fi
  fi

  if [[ "$commit_run_artifacts" == "1" ]]; then
    bash scripts/h200_commit_run_artifacts.sh "$results_dir" "$branch"
  fi
}

merge_stage_results() {
  local run_stage="$1"
  local output_dir="results/selectivity_h200_${run_stage}_combined"
  local generated_dir="docs/generated/selectivity_h200_${run_stage}_combined"
  local -a merge_args=()
  local key
  for key in "${model_keys[@]}"; do
    local run_dir="results/selectivity_h200_${run_stage}_${key}"
    if [[ -f "$run_dir/generations.jsonl" ]]; then
      merge_args+=(--run-dir "$run_dir")
    fi
  done
  if [[ "${#merge_args[@]}" -eq 0 ]]; then
    echo "No completed ${run_stage} selectivity runs to merge." >&2
    return 0
  fi

  uv run python scripts/merge_selectivity_panel_results.py \
    --output-dir "$output_dir" \
    "${merge_args[@]}"
  uv run python scripts/make_figures.py --results-dir "$output_dir"
  uv run python scripts/export_paper_assets.py --results-dir "$output_dir" --paper-dir "$generated_dir"
  uv run python scripts/report_selectivity_status.py --run-dir "$output_dir"
  uv run python scripts/plan_ci_power.py \
    --results-dir "$output_dir" \
    --target-ci-width "$target_ci_width" \
    --output-json "$output_dir/ci_power.json" \
    --output-md "$generated_dir/ci_power.md"

  if [[ "$commit_run_artifacts" == "1" ]]; then
    bash scripts/h200_commit_run_artifacts.sh "$output_dir" "$branch"
  fi
}

if [[ "$stage" == "powered" || "$stage" == "all" ]]; then
  register_power_plan
  prepare_powered_data
fi

stages=()
if [[ "$stage" == "all" ]]; then
  stages=(smoke powered)
else
  stages=("$stage")
fi

for run_stage in "${stages[@]}"; do
  for key in "${model_keys[@]}"; do
    run_config "$run_stage" "$key"
    echo "Waiting for GPU to clear after selectivity ${run_stage} ${key}..."
    bash scripts/wait_for_h200_gpu.sh
  done
  merge_stage_results "$run_stage"
done

echo "Selectivity panel ${stage} run complete for model keys: ${model_keys[*]}"
