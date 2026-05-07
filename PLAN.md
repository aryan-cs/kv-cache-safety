# KV-Cache Safety Selectivity Execution Plan

## Current State

The project is pivoting from a Qwen-centered cache-erasure draft to a cross-family safety-selectivity study. Existing Qwen H200 artifacts are preserved as pilot evidence and will not be relabeled as final cross-family evidence. The active experimental path is now the protocol in `RESEARCH.md`.

The H200 instance was not reachable at the start of this implementation pass. `ssh uiuc-h200` returned a JupyterHub `302 Found` redirect and closed the connection. Remote cleanup, remote installation checks, and H200 smoke runs are blocked until the instance is restarted.

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
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | Gated; requires accepted access on H200. |
| Gemma | `google/gemma-2-9b-it` | Gated; requires accepted access on H200. |
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
3. Verify no active run lock/process before launching.
4. Run smoke:

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/selectivity_h200_qwen2_5_7b_instruct_smoke.yaml \
  --resume
```

5. Watch progress:

```bash
uv run python scripts/report_selectivity_status.py \
  --run-dir results/selectivity_h200_qwen2_5_7b_instruct_smoke
```

6. Commit and push run artifacts from H200:

```bash
bash scripts/h200_commit_run_artifacts.sh results/selectivity_h200_qwen2_5_7b_instruct_smoke master
```

7. Fetch to Mac. If the Mac worktree is too dirty to pull safely, use the rsync fetch helper:

```bash
bash scripts/fetch_h200_selectivity_results.sh results/selectivity_h200_qwen2_5_7b_instruct_smoke
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

The implemented harness:

```bash
uv run python scripts/judge_with_codex_gemini.py \
  --input-jsonl results/<run_id>/generations.jsonl \
  --output-jsonl results/<run_id>/judgments.codex_gemini.jsonl \
  --providers codex,gemini \
  --workers 2 \
  --resume
```

The judge harness preserves raw output, raw-output hash, prompt hash, rubric hash, command, parser status, timestamps, and label source type. These are model-judge labels, not human labels.

## Power And Decision Gates

Before powered Phase 2 runs:

- compute cell counts needed for `SSEI_abs` and `SSEI_logodds` precision;
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

## Immediate Implementation Checklist

- Add `user_pinned` cache policy support.
- Add restart-safe `cache_stats.jsonl` and `progress.json`.
- Add local Codex/Gemini judge harness.
- Add H200 artifact commit helper.
- Add Mac fetch helper.
- Add selectivity smoke configs.
- Run local tests.
- Retry H200 SSH after restart.
- Run H200 Qwen 2.5 7B instruct smoke.
- Sync H200 data back to Mac.
- Run local judging smoke on synced data.
