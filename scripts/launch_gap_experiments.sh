#!/usr/bin/env bash
# Launch all gap-closing experiments on H200 in tmux sessions.
# Run this script ON the H200 after pushing code.
#
# Usage: bash scripts/launch_gap_experiments.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Gap 1: Expanded base alignment contrast ==="
echo "Running base model (Qwen2.5-7B) with full public_refusal_safety suite"
tmux new-session -d -s gap1_base_alignment \
    "cd $REPO_ROOT && uv run python scripts/run_experiment.py --config configs/experiments/selectivity_h200_base_alignment_expanded.yaml --resume 2>&1 | tee logs/gap1_base_alignment.log"

echo "=== Gap 2: Budget sweep (3 models x 3 budgets) ==="
echo "Running Qwen2.5-14B budget sweep [64, 256, 512]"
tmux new-session -d -s gap2_budget_qwen14b \
    "cd $REPO_ROOT && uv run python scripts/run_experiment.py --config configs/experiments/selectivity_h200_budget_sweep_qwen2_5_14b_instruct.yaml --resume 2>&1 | tee logs/gap2_budget_qwen14b.log"

echo "Running Phi-4 budget sweep [64, 256, 512]"
tmux new-session -d -s gap2_budget_phi4 \
    "cd $REPO_ROOT && uv run python scripts/run_experiment.py --config configs/experiments/selectivity_h200_budget_sweep_phi4.yaml --resume 2>&1 | tee logs/gap2_budget_phi4.log"

echo "Running Llama-3.1-8B budget sweep [64, 256, 512]"
tmux new-session -d -s gap2_budget_llama \
    "cd $REPO_ROOT && uv run python scripts/run_experiment.py --config configs/experiments/selectivity_h200_budget_sweep_llama3_1_8b_instruct.yaml --resume 2>&1 | tee logs/gap2_budget_llama.log"

echo "=== Gap 3: Llama causal patching (quantization probe) ==="
echo "Running Llama-3.1-8B causal patching with kv_int4_sim"
tmux new-session -d -s gap3_llama_causal \
    "cd $REPO_ROOT && uv run python scripts/run_experiment.py --config configs/experiments/h200_causal_patch_llama3_1_8b_instruct.yaml --resume 2>&1 | tee logs/gap3_llama_causal.log"

echo ""
echo "All experiments launched. Monitor with:"
echo "  tmux ls                    # list sessions"
echo "  tmux attach -t gap1_base_alignment  # attach to a session"
echo "  tail -f logs/gap*.log      # follow logs"
echo ""
echo "Expected GPU usage:"
echo "  gap1 + gap2_qwen14b: ~7B + ~14B params = ~42GB"
echo "  gap2_phi4: ~14B params = ~28GB"
echo "  gap2_llama + gap3_llama: ~8B params = ~16GB (run sequentially via tmux)"
echo ""
echo "NOTE: GPU memory may not allow all sessions simultaneously."
echo "If OOM, run gap2_qwen14b first, then gap2_phi4 + gap2_llama sequentially."
