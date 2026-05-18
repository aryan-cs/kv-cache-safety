#!/usr/bin/env bash
# Integrate gap-closing experiment results from H200.
# Pulls results, runs analysis, rebuilds PDF.
#
# Usage: bash scripts/integrate_gap_results.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
H200="uiuc-h200"
REMOTE_ROOT="~/sandbox/llm-safety"

echo "=== Step 1: Pull results from H200 ==="

# Base alignment expanded (replaces old 2-prompt base model)
echo "Pulling base alignment expanded..."
BASE_DIR=$(ssh $H200 "ls -d $REMOTE_ROOT/results/selectivity_h200_base_alignment_expanded_*/ 2>/dev/null | head -1" | tr -d '/')
if [ -n "$BASE_DIR" ]; then
    BASE_NAME=$(basename "$BASE_DIR")
    rsync -avz --progress "$H200:$REMOTE_ROOT/results/$BASE_NAME/" "results/selectivity_h200_powered_qwen2_5_7b_base/" \
        --exclude='*.lock' --exclude='cache_stats.jsonl'
    echo "  Pulled base alignment -> selectivity_h200_powered_qwen2_5_7b_base"
else
    echo "  WARNING: No base alignment results found on H200"
fi

# Llama causal patching
echo "Pulling Llama causal patching..."
LLAMA_CAUSAL_DIR=$(ssh $H200 "ls -d $REMOTE_ROOT/results/h200_causal_patch_llama3_1_8b_instruct_*/ 2>/dev/null | head -1" | tr -d '/')
if [ -n "$LLAMA_CAUSAL_DIR" ]; then
    LLAMA_CAUSAL_NAME=$(basename "$LLAMA_CAUSAL_DIR")
    rsync -avz --progress "$H200:$REMOTE_ROOT/results/$LLAMA_CAUSAL_NAME/" "results/h200_causal_patch_llama3_1_8b_instruct/" \
        --exclude='*.lock' --exclude='cache_stats.jsonl'
    echo "  Pulled Llama causal -> h200_causal_patch_llama3_1_8b_instruct"
else
    echo "  WARNING: No Llama causal results found on H200"
fi

# Budget sweeps
for model in llama3_1_8b_instruct phi4 qwen2_5_14b_instruct; do
    echo "Pulling $model budget sweep..."
    SWEEP_DIR=$(ssh $H200 "ls -d $REMOTE_ROOT/results/selectivity_h200_budget_sweep_${model}_*/ 2>/dev/null | head -1" | tr -d '/')
    if [ -n "$SWEEP_DIR" ]; then
        SWEEP_NAME=$(basename "$SWEEP_DIR")
        rsync -avz --progress "$H200:$REMOTE_ROOT/results/$SWEEP_NAME/" "results/selectivity_h200_budget_sweep_${model}/" \
            --exclude='*.lock' --exclude='cache_stats.jsonl'
        echo "  Pulled $model budget sweep"
    else
        echo "  WARNING: No $model budget sweep results found on H200"
    fi
done

echo ""
echo "=== Step 2: Run analysis scripts ==="

echo "Running claim assessment..."
uv run python scripts/make_selectivity_claim_assessment.py

echo "Running cross-model summary..."
uv run python scripts/make_cross_model_summary.py

echo "Running budget dose-response..."
uv run python scripts/make_budget_dose_response.py

echo "Running family replication table..."
uv run python scripts/make_family_replication_table.py

echo "Running robustness analysis..."
uv run python scripts/make_robustness_analysis.py

echo ""
echo "=== Step 3: Build PDF ==="
cd docs/latex
tectonic main.tex
cd "$REPO_ROOT"
cp docs/latex/main.pdf docs/kv-cache-safety.pdf 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Check docs/generated/claim_assessment/claim_assessment.json for claim status"
echo "PDF at docs/kv-cache-safety.pdf"
