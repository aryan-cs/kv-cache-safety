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
- run id: latest local `tiny_hf_smoke_*`
- machine / GPU: local development environment
- model: `sshleifer/tiny-gpt2`
- prompt suites: `capability_smoke`
- cache policies: `none`, `sliding_window`, `kv_int8_sim`
- main metrics: not meaningful
- decision: discard as evidence; keep only as plumbing validation
