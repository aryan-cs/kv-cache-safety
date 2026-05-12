#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
audit_per_suite_policy="${AUDIT_PER_SUITE_POLICY:-10}"
audit_annotator_template_count="${AUDIT_ANNOTATOR_TEMPLATE_COUNT:-2}"
audit_output_dir="${AUDIT_OUTPUT_DIR:-docs/audit}"

require_completed_results() {
  local results_dir="$1"
  for required in manifest.json generations.jsonl metrics.json; do
    if [[ ! -f "$results_dir/$required" ]]; then
      echo "Missing required completed-run artifact: $results_dir/$required" >&2
      echo "Fetch and aggregate the completed H200 run before exporting audit samples." >&2
      exit 1
    fi
  done
}

export_audit_sample() {
  local results_dir="$1"
  require_completed_results "$results_dir"
  uv run python scripts/export_human_audit_sample.py \
    --results-dir "$results_dir" \
    --output-dir "$audit_output_dir" \
    --per-suite-policy "$audit_per_suite_policy" \
    --annotator-template-count "$audit_annotator_template_count" \
    --include-hidden-reference
}

export_audit_sample "$primary_results"
export_audit_sample "$causal_results"

echo "Publication audit samples exported to $audit_output_dir"
echo "Complete the annotator CSVs, then aggregate with scripts/aggregate_human_audit.py."
