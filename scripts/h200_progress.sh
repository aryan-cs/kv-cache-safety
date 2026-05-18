#!/usr/bin/env bash
data=$(ssh uiuc-h200 'cat ~/sandbox/llm-safety/results/selectivity_h200_powered_qwen2_5_14b_msm_value_aug*/progress.json' 2>/dev/null)
if [ -z "$data" ]; then
  echo "No data (SSH failed or experiment finished)"
  exit 0
fi
echo "$data" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['current']
    print(f\"{d['completed']}/{d['expected']} ({d['progress_percent']:.1f}%) — {c['policy']} / {c['suite']}\")
except Exception as e:
    print(f'Parse error: {e}')
"
