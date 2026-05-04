# Experiment Log

Use this file to summarize meaningful runs after aggregation.

Required fields for each entry:

- date
- commit hash
- config path
- run id
- machine / GPU
- model
- prompt suites
- cache policies
- main metrics
- decision: keep, rerun, discard, or extend

## 2026-05-03 Local Plumbing Runs

- date: 2026-05-03
- commit hash: local dirty tree during development; do not cite
- config path: `configs/experiments/tiny_hf_smoke.yaml`
- run id: `tiny_hf_guard_smoke_20260503`
- machine / GPU: local development environment
- model: `sshleifer/tiny-gpt2`
- prompt suites: `capability_smoke`
- cache policies: `none`, `sliding_window`, `kv_int8_sim`
- main metrics: not meaningful; artifact/readiness plumbing produced `generations.jsonl`, `metrics.json`, `cache_stats.parquet`, and figures with explicit tiny/smoke override flags
- decision: discard as evidence; keep only as plumbing validation

## 2026-05-04 H200 Launcher

- date: 2026-05-04
- commit hash: `d3670cc`
- config path: `scripts/run_h200_sweep.sh`
- run id: pending; launcher waits for `h200_qwen_full_sweep` and `h200_causal_patch_qwen7b`
- machine / GPU: UIUC H200 notebook
- model: pending; Qwen2.5 sweep has not started
- prompt suites: pending; public-suite preparation has not started in the launcher
- cache policies: pending; sweep has not started
- main metrics: no empirical metrics yet; `scripts/wait_and_run_h200_sweep.sh` passed ruff and the full test suite, then entered the GPU gate because the H200 was saturated
- decision: keep launcher waiting; do not cite as evidence
