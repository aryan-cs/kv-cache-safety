# Registered Selectivity Panel Runbook

This is the handoff procedure for the `RESEARCH.md` selectivity protocol. It assumes a fresh clone of `master`, an authorized Illinois Computes H200 notebook allocation, and Hugging Face access for every model you choose to run.

## What This Runs

The registered panel is generated from `configs/models/selectivity_panel.yaml` by `scripts/generate_selectivity_configs.py`.

Default non-gated model keys:

- `gpt_oss_20b`
- `qwen2_5_7b_base`
- `qwen2_5_7b_instruct`
- `qwen3_5_9b`
- `mistral_7b_instruct_v0_3`
- `olmo3_7b_instruct`
- `phi4`

Set `SELECTIVITY_INCLUDE_GATED=1` only when the H200 environment has accepted Hugging Face access for `llama3_1_8b_instruct` and `gemma2_9b_it`. The launcher skips configs whose pinned revision is unavailable, but skipped gated models do not count as cross-family evidence.

## Fresh Local Setup

```bash
git clone git@github.com:aryan-cs/kv-cache-safety.git llm-safety
cd llm-safety
git checkout master
git pull --ff-only origin master
uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q
```

Initialize or refresh the H200 checkout under the path enforced by the guarded launcher:

```bash
bash scripts/setup_h200_remote.sh
```

The remote checkout must be clean and located at `${H200_WORKDIR:-/home/aryang9/sandbox/llm-safety}`. `scripts/wait_and_run_h200_sweep.sh` refuses any other path so that result manifests point to a known repository location.

## Phase 0 Freeze And Power Plan

Before powered runs, inspect the frozen panel and power plan:

```bash
uv run python scripts/generate_selectivity_configs.py --stage smoke --overwrite
uv run python scripts/generate_selectivity_configs.py --stage powered --overwrite
uv run python scripts/plan_ci_power.py \
  --target-ci-width "${TARGET_CI_WIDTH:-0.08}" \
  --output-json docs/generated/selectivity_panel_phase0_ci_power.json \
  --output-md docs/generated/selectivity_panel_phase0_ci_power.md
```

If the power plan cannot resolve the registered SSEI thresholds for a suite or family, mark that analysis exploratory before launching powered generation. Do not reinterpret an underpowered confirmatory run after seeing the outputs.

## Launch On H200

From the H200 checkout:

```bash
cd /home/aryang9/sandbox/llm-safety
SELECTIVITY_STAGE=all \
SWEEP_SCRIPT=scripts/run_h200_selectivity_panel.sh \
setsid -f bash scripts/wait_and_run_h200_sweep.sh \
  </dev/null > logs/h200/selectivity_launcher.out 2>&1
```

Useful launch controls:

- `SELECTIVITY_STAGE=smoke`, `powered`, or `all`.
- `SELECTIVITY_MODELS="qwen2_5_7b_instruct mistral_7b_instruct_v0_3"` to run a subset by registered key.
- `SELECTIVITY_INCLUDE_GATED=1` to include Llama/Gemma after HF access is approved.
- `PUBLIC_PROMPT_LIMIT=650` controls powered public-suite prompt count.
- `TARGET_CI_WIDTH=0.08` controls readiness and power reports.
- `COMMIT_RUN_ARTIFACTS=1` commits and pushes each completed `results/...` directory with its matching `docs/generated/...` assets to `master`.

The guarded launcher syncs `master`, runs `ruff`, runs the full test suite, waits for the H200 GPU gate, syncs and validates again, then starts the selectivity panel. `run_h200_selectivity_panel.sh` refuses a dirty worktree.

## Expected Outputs

Per-model outputs:

- `results/selectivity_h200_<stage>_<model_key>/`
- `docs/generated/selectivity_h200_<stage>_<model_key>/`

Combined stage outputs:

- `results/selectivity_h200_<stage>_combined/`
- `docs/generated/selectivity_h200_<stage>_combined/`

Phase 0 power documentation:

- `docs/generated/selectivity_panel_phase0_ci_power.json`
- `docs/generated/selectivity_panel_phase0_ci_power.md`

Each completed result directory contains `manifest.json`, `config.resolved.yaml`, `environment.json`, `prompts.jsonl`, `generations.jsonl`, `metrics.json`, `cache_stats.parquet`, and generated figures. The matching `docs/generated/...` directory contains tables, macros, CI reports, and an artifact manifest.

## Fetch And Verify Results

From a clean local checkout, fetch all completed selectivity artifacts for the selected stage and model list:

```bash
SELECTIVITY_STAGE=all bash scripts/fetch_h200_selectivity_panel.sh
```

Use the same `SELECTIVITY_MODELS` and `SELECTIVITY_INCLUDE_GATED` values that were used for the run. The fetch wrapper discovers completed remote artifacts, delegates to `scripts/fetch_h200_results.sh`, writes local and remote manifests under `logs/h200/`, and compares hashes before returning success.

To fetch one explicit artifact set instead:

```bash
bash scripts/fetch_h200_results.sh \
  results/selectivity_h200_powered_qwen2_5_7b_instruct \
  docs/generated/selectivity_h200_powered_qwen2_5_7b_instruct \
  results/selectivity_h200_powered_combined \
  docs/generated/selectivity_h200_powered_combined
```

## Audit And Judge Diagnostics

Export blinded audit sheets from the combined powered panel or from any selected per-model run:

```bash
uv run python scripts/export_human_audit_sample.py \
  --results-dir results/selectivity_h200_powered_combined \
  --output-dir docs/audit \
  --per-suite-policy 10 \
  --annotator-template-count 2 \
  --include-hidden-reference
```

Complete the human annotator CSVs, then aggregate them:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv docs/audit/selectivity_h200_powered_combined_audit_blinded_annotator_*.csv \
  --key-jsonl docs/audit/selectivity_h200_powered_combined_audit_key.jsonl \
  --results-dir results/selectivity_h200_powered_combined \
  --export-manifest docs/audit/selectivity_h200_powered_combined_audit_export_manifest.json \
  --output-dir docs/audit/selectivity_h200_powered_combined_summary
```

Open/local judge labels are diagnostic. If you run them, preserve the output source separately and do not present them as human evidence.

External Gemini judge diagnostics require explicit egress approval on the blinded audit JSONL. Gemini is the only supported local judge provider for this project.

```bash
uv run python scripts/approve_judge_egress.py \
  --input-jsonl docs/audit/selectivity_h200_powered_combined_audit_sample.jsonl \
  --output-jsonl docs/audit/selectivity_h200_powered_combined_audit_sample_egress_approved.jsonl \
  --approval-note "Approved blinded selectivity audit rows for external judge diagnostics."

uv run python scripts/judge_with_gemini.py \
  --input-jsonl docs/audit/selectivity_h200_powered_combined_audit_sample_egress_approved.jsonl \
  --output-jsonl docs/audit/selectivity_h200_powered_combined_gemini_judge.jsonl \
  --providers gemini \
  --judge-mode all-providers \
  --allow-data-egress \
  --resume
```

## Claim Gates And Paper Build

The selectivity panel can support behavioral selectivity, cross-family replication, mitigation, and base-model alignment-contrast claims when the registered gates pass. The stronger cache-mediated safety-erasure claim still requires Phase 4 causal diagnostics: run cache restoration or patching only for families with meaningful Phase 2 SSEI effects, compare `policy_pinned` against matched `user_pinned`, and feed those causal result directories to `scripts/assess_claims.py`.

For the existing registered Qwen causal extension wrapper, launch through the same guarded H200 gate:

```bash
SWEEP_SCRIPT=scripts/run_h200_causal_ci_extension.sh \
bash scripts/wait_and_run_h200_sweep.sh
```

For the existing Qwen causal paper pipeline, the publication build remains:

```bash
bash scripts/build_publication_artifacts.sh
```

That wrapper is fail-closed: it requires completed primary and causal result artifacts, completed audit summaries, passing evidence-gated claim assessment, a valid PDF, and an arXiv source bundle. For selectivity-panel papers, pass the selected primary/causal result and generated directories explicitly through the same script interfaces once the causal diagnostics exist.

Summarize remaining blockers without mutating artifacts:

```bash
uv run python scripts/report_publication_status.py \
  --allow-missing-paper-pdf \
  --output-json docs/build/publication_status.json \
  --output-md docs/build/publication_status.md
```

## Known External Blockers

- H200/JupyterHub SSH access must work before any remote run can start.
- Gated Hugging Face models require accepted licenses and a valid token in the H200 environment.
- Human audit labels cannot be generated by code; the code exports, blinds, validates, and aggregates them.
- Phase 4 causal diagnostics are conditional on Phase 2 effects and are not automatically launched by the selectivity panel.
