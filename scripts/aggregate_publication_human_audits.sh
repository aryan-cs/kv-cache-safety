#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_run_id="${PRIMARY_RUN_ID:-h200_qwen_full_sweep}"
causal_run_id="${CAUSAL_RUN_ID:-h200_causal_patch_qwen7b}"
primary_results="${PRIMARY_RESULTS_DIR:-results/$primary_run_id}"
causal_results="${CAUSAL_RESULTS_DIR:-results/$causal_run_id}"
audit_input_dir="${AUDIT_INPUT_DIR:-paper/audit}"
primary_output_dir="${PRIMARY_AUDIT_SUMMARY_DIR:-paper/audit/${primary_run_id}_summary}"
causal_output_dir="${CAUSAL_AUDIT_SUMMARY_DIR:-paper/audit/${causal_run_id}_summary}"

require_file() {
  local path="$1"
  local message="$2"
  if [[ ! -f "$path" ]]; then
    echo "$message: $path" >&2
    exit 1
  fi
}

require_completed_results() {
  local results_dir="$1"
  for required in manifest.json generations.jsonl metrics.json; do
    require_file \
      "$results_dir/$required" \
      "Missing required completed-run artifact; run scripts/prepare_after_h200_fetch.sh first"
  done
}

aggregate_run_audit() {
  local run_id="$1"
  local results_dir="$2"
  local output_dir="$3"
  local key_jsonl="$audit_input_dir/${run_id}_audit_key.jsonl"
  local export_manifest="$audit_input_dir/${run_id}_audit_export_manifest.json"

  require_completed_results "$results_dir"
  require_file "$key_jsonl" "Missing audit key JSONL"
  require_file "$export_manifest" "Missing audit export manifest"

  shopt -s nullglob
  local open_judge_csvs=("$audit_input_dir/${run_id}_audit_blinded_annotator_open_judge_"*.csv)
  local audit_csvs=()
  if [[ "${#open_judge_csvs[@]}" -gt 0 ]]; then
    audit_csvs=("${open_judge_csvs[@]}")
  else
    audit_csvs=("$audit_input_dir/${run_id}_audit_blinded_annotator_"*.csv)
  fi
  shopt -u nullglob
  if [[ "${#audit_csvs[@]}" -eq 0 ]]; then
    echo "Missing completed annotator CSVs: $audit_input_dir/${run_id}_audit_blinded_annotator_*.csv" >&2
    exit 1
  fi

  uv run python scripts/aggregate_human_audit.py \
    --audit-csv "${audit_csvs[@]}" \
    --key-jsonl "$key_jsonl" \
    --results-dir "$results_dir" \
    --export-manifest "$export_manifest" \
    --output-dir "$output_dir"
  uv run python scripts/check_human_audit_readiness.py \
    --summary-json "$output_dir/human_audit_summary.json" \
    --audit-manifest "$output_dir/audit_manifest.json" \
    --results-dir "$results_dir" \
    --require-result-source-match \
    --require-baseline-deltas
}

uv sync --frozen --extra dev
aggregate_run_audit "$primary_run_id" "$primary_results" "$primary_output_dir"
aggregate_run_audit "$causal_run_id" "$causal_results" "$causal_output_dir"
uv run python scripts/post_h200_next_steps.py \
  --output-json paper/generated/post_h200_next_steps.json \
  --output-md paper/generated/post_h200_next_steps.md

echo "Publication human audits aggregated:"
echo "- $primary_output_dir"
echo "- $causal_output_dir"
