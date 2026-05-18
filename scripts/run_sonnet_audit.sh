#!/usr/bin/env bash
# Run Claude Sonnet judging on all 12 panel models for audit provenance.
# This stamps data_egress_approved on public-dataset safety rows, then
# calls judge_with_gemini.py (which supports --providers claude despite the name).
#
# Usage: bash scripts/run_sonnet_audit.sh [--workers N] [--limit N]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/results"
AUDIT_DIR="$REPO_ROOT/docs/audit"
SCRIPTS_DIR="$REPO_ROOT/scripts"

WORKERS="${1:-2}"
LIMIT="${2:-}"

PUBLIC_SAFETY_SUITES="public_refusal_safety|public_system_leakage|public_benign_overrefusal|public_xstest_safe"

mkdir -p "$AUDIT_DIR"

for run_dir in "$RESULTS_ROOT"/selectivity_h200_powered_*; do
    [ -d "$run_dir" ] || continue
    [ "$(basename "$run_dir")" = "selectivity_h200_powered_combined" ] && continue

    model_key="${run_dir##*selectivity_h200_powered_}"
    gen_file="$run_dir/generations.jsonl"
    [ -f "$gen_file" ] || continue

    # Create egress-stamped version with only public safety suites
    stamped_file="$run_dir/generations.egress_stamped.jsonl"
    echo "Stamping $model_key ..."
    python3 -c "
import json, sys, re
pattern = re.compile(r'^($PUBLIC_SAFETY_SUITES)$')
with open('$gen_file') as fin, open('$stamped_file', 'w') as fout:
    for line in fin:
        row = json.loads(line)
        suite = row.get('suite', '')
        if pattern.match(suite):
            row['data_egress_approved'] = True
            fout.write(json.dumps(row, sort_keys=True) + '\n')
"

    row_count=$(wc -l < "$stamped_file" | tr -d ' ')
    echo "  $model_key: $row_count public safety rows to judge"

    if [ "$row_count" -eq 0 ]; then
        echo "  Skipping $model_key (no public safety rows)"
        continue
    fi

    output_file="$AUDIT_DIR/selectivity_h200_powered_${model_key}_judgments.claude.jsonl"

    limit_args=""
    if [ -n "$LIMIT" ]; then
        limit_args="--limit $LIMIT"
    fi

    echo "  Judging $model_key with Claude Sonnet ..."
    cd "$REPO_ROOT"
    uv run python "$SCRIPTS_DIR/judge_with_gemini.py" \
        --input-jsonl "$stamped_file" \
        --output-jsonl "$output_file" \
        --providers claude \
        --claude-model claude-sonnet-4-6 \
        --workers "$WORKERS" \
        --allow-data-egress \
        --resume \
        $limit_args || echo "  WARNING: judging failed for $model_key"

    echo "  Done: $model_key"
done

echo "All models judged. Results in $AUDIT_DIR"
