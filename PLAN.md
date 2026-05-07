# KV-Cache Safety Selectivity Execution Plan

For an executable fresh-clone handoff, use `SELECTIVITY_RUNBOOK.md`. This file is a planning log and may mention pilot artifacts or one-off smoke commands that are not the current registered launch path.

## Current State

The project is pivoting from a Qwen-centered cache-erasure draft to a cross-family safety-selectivity study. Existing Qwen H200 artifacts are preserved as pilot evidence and will not be relabeled as final cross-family evidence. The active experimental path is now the protocol in `RESEARCH.md`.

Model access is Hugging Face only. Do not use Ollama for this protocol.

Hugging Face gated access is working on H200 with the supplied read-only token. The smoke matrix has completed for the public and gated models that cleared harness validation: GPT-OSS 20B, Qwen 2.5 base, Qwen 2.5 Instruct, Llama 3.1 8B Instruct, Gemma 2 9B IT, Mistral 7B Instruct v0.3, OLMo 3 7B Instruct, and Phi-4. Chat-safety smoke outputs have been judged locally with Codex/Gemini where family separation permits, with Gemini-only judging for GPT-OSS.

The active H200 run is the first powered sweep:

```text
selectivity_h200_powered_qwen2_5_7b_instruct
```

It uses 1,300 public prompts per confirmatory suite with the full eviction/retention policy matrix. The run writes `generations.jsonl`, `cache_stats.jsonl`, and `progress.json` incrementally and is restartable with `--resume`.

The first 225 rows were generated at commit `d7b4431`. The run was then paused and resumed at commit `90ca99e`, which preserves generation semantics but limits expensive cache L2 norm diagnostics to pre-response cache states by default. This keeps the confirmatory outcomes unchanged while making the powered sweep operationally tractable. The mixed diagnostic metadata is provenance-relevant and must be reported if cache-norm figures use this run.

## What "Clear Out The Codebase" Means

Do not delete salvageable data. The old pipeline is retired from the active research path by:

- keeping existing `results/` and `docs/audit/` artifacts as pilot archives;
- adding new selectivity-specific configs, scripts, and plan files;
- avoiding reuse of old `h200_*` run IDs for the new study;
- marking old Qwen-only evidence as pilot material in analysis and paper text;
- using new run names beginning with `selectivity_`.

When H200 is reachable, perform the same cleanup there: preserve completed result directories and paper/audit artifacts, but launch only new `selectivity_*` runs.

## Scientific Goal

The paper should not claim that KV-cache compression can degrade safety as a new observation. Prior work already establishes that cache compression/eviction can cause safety and security failures. The contribution is:

1. formal selectivity measurement with `SSEI`;
2. cross-family replication of selectivity;
3. role-aware mitigation with `policy_pinned`;
4. causal localization against a matched `user_pinned` control;
5. restartable, provenance-complete experiments.

## Model Matrix

Primary chat-safety models:

| Family | Checkpoint | Status |
| --- | --- | --- |
| GPT-OSS | `openai/gpt-oss-20b` | Public/open on Hugging Face; use as evaluated model only if H200 harness validation passes. |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct` | Pinned smoke config added. |
| Qwen 3.5 | `Qwen/Qwen3.5-9B` | Exists, but HF metadata is multimodal; use only if text-only cache harness passes. |
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | Gated HF access validated; smoke run complete and locally judged. |
| Gemma | `google/gemma-2-9b-it` | Gated HF access validated by preflight; run next after H200 restart. |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | Public Apache-2.0; good fallback/additional family. |
| OLMo | `allenai/Olmo-3-7B-Instruct` | Public/open; keep current ID. |
| Phi | `microsoft/phi-4` | Public; `microsoft/Phi-4-mini-instruct` is a fallback if needed. |

Base-model control:

| Family | Checkpoint | Role |
| --- | --- | --- |
| Qwen 2.5 | `Qwen/Qwen2.5-7B` | Base-model alignment contrast. Do not force chat-refusal metrics unless a frozen scaffold is registered. |

## Interventions

Primary eviction/retention matrix:

- `none`
- `sliding_window`
- `sink_recent`
- `random_matched`
- `policy_pinned`
- `user_pinned`

`policy_pinned` protects system/policy tokens. `user_pinned` protects user tokens at the same retained-token budget and is the matched non-policy control needed for H5.

`kv_int8_sim` and `kv_int4_sim` are not primary eviction policies. They are Phase 5 perturbation diagnostics and must be reported separately from eviction results.

## Restartability And Data Safety

Generation runs must be restartable after H200 expiration:

- write `generations.jsonl` one row at a time;
- write durable `cache_stats.jsonl` one decision row at a time;
- write `progress.json` after each completed prompt/policy/seed row;
- support `--resume` without rerunning completed rows;
- quarantine corrupt JSONL tails rather than silently dropping data;
- reconcile generation rows against cache-stat evidence on resume;
- preserve `manifest.json`, `environment.json`, `config.resolved.yaml`, prompt manifests, and cache statistics.

The current implementation adds durable `cache_stats.jsonl` and `progress.json` to the existing runner.

## H200 Workflow

1. Restart H200 if SSH fails.
2. On H200, work only in `/home/aryang9/sandbox/llm-safety`.
3. Use Hugging Face token/cache discovery. Do not set `HF_HOME` unless intentionally overriding the cache path.
4. Verify no active run lock/process before launching.
5. Run a powered model sweep:

```bash
SELECTIVITY_STAGE=powered SELECTIVITY_MODELS=qwen2_5_7b_instruct \
  bash scripts/run_h200_selectivity_panel.sh
```

6. Watch progress:

```bash
uv run python scripts/report_selectivity_status.py \
  --run-dir results/selectivity_h200_powered_qwen2_5_7b_instruct
```

7. Commit and push run artifacts from H200:

```bash
bash scripts/h200_commit_run_artifacts.sh \
  results/selectivity_h200_powered_qwen2_5_7b_instruct master
```

8. Fetch to Mac. If the Mac worktree is too dirty to pull safely, use the tar fetch helper:

```bash
bash scripts/fetch_h200_selectivity_results.sh \
  results/selectivity_h200_powered_qwen2_5_7b_instruct
```

## Local Judging Workflow

Judging happens on the Mac, not H200.

Primary judge command:

```bash
codex exec --cd /Users/aryan/Desktop/projects/llm-safety --sandbox read-only --ephemeral --output-last-message <tmpfile> -
```

Fallback judge command:

```bash
gemini -p '<prompt>'
```

For powered runs, judge sampled audit rows rather than blindly judging every generation row. Export a per-model audit sample first:

```bash
uv run python scripts/export_human_audit_sample.py \
  --results-dir results/<run_id> \
  --output-dir paper/audit \
  --per-suite-policy 25 \
  --strategy effect \
  --annotator-template-count 2 \
  --include-hidden-reference \
  --seed 0
```

For merged panel artifacts, the audit exporter preserves `source_run_id` and `model_id` scope so identical public prompt IDs from different models cannot collapse into one sampled item.

Then approve the key JSONL for local model-judge data egress and run the external judge harness:

```bash
uv run python scripts/approve_judge_egress.py \
  --input-jsonl paper/audit/<run_id>_audit_key.jsonl \
  --output-jsonl results/<run_id>/judge_input.codex_gemini.jsonl \
  --approval-note "User approved local Codex/Gemini judging for H200-generated selectivity rows." \
  --approval-source user_instruction \
  --overwrite

uv run python scripts/judge_with_codex_gemini.py \
  --input-jsonl results/<run_id>/judge_input.codex_gemini.jsonl \
  --output-jsonl results/<run_id>/judgments.codex_gemini.jsonl \
  --providers codex,gemini \
  --judge-mode all-providers \
  --workers 4 \
  --resume \
  --allow-data-egress
```

The judge harness preserves raw output, raw-output hash, prompt hash, rubric hash, command, parser status, timestamps, response-length bucket, data-egress approval provenance, and label source type. These are model-judge labels, not human labels.

## Power And Decision Gates

Before powered Phase 2 runs:

- compute cell counts needed for `SSEI_abs` and `SSEI_logodds` precision;
- use at least `1201` public prompt clusters per confirmatory suite by default; the launcher currently requests `1300` to leave headroom above the conservative two-component SSEI planning count for a full CI width of `0.08`;
- mark underpowered suites exploratory before launch;
- require at least two instruction-tuned families with positive `SSEI` for cross-family selectivity;
- require `policy_pinned` to beat `sink_recent` for mitigation;
- require `policy_pinned` to beat `user_pinned` for causal localization.

## Heartbeat Status Format

Every heartbeat/update should use:

```text
[What's currently being run] Progress: X%
[Description of what the test does for this paper in simple terms]
Currently [doing/testing/etc] [Model name/process/activity]
Estimated Time Remaining: [Estimated time left to finish this specific activity based on timestamps of previous checkpoint messages & progress rate] minutes
```

Use `scripts/report_selectivity_status.py` to generate this format from a run directory.

## Active Implementation Checklist

- Monitor `selectivity_h200_powered_qwen2_5_7b_instruct` until completion or H200 expiry.
- If H200 expires, notify Aryan and resume the same run after restart; do not relaunch a duplicate run while a process or lock is active.
- Fetch and push completed powered Qwen artifacts.
- Export powered audit samples per model, approve only those sampled rows for local judge egress, and run Codex/Gemini judging with family separation.
- Launch the next powered model while the previous model's local judging is running.
- Merge completed powered runs only after each per-model run has valid artifacts and readiness reports.
- Extend paper analysis only from completed, provenance-valid runs; keep smoke and pilot evidence scoped accordingly.
