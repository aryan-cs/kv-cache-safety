# KV-Cache Safety Selectivity Execution Plan

For an executable fresh-clone handoff, use `SELECTIVITY_RUNBOOK.md`. This file is a planning log and may mention pilot artifacts or one-off smoke commands that are not the current registered launch path.

## Current State

The active project is a cross-family safety-selectivity study under the protocol in `RESEARCH.md`. Existing older Qwen-centered artifacts are preserved as pilot or historical evidence and must not be relabeled as final cross-family evidence.

Model access is Hugging Face only. Do not use Ollama for this protocol.

Hugging Face public and gated access is working on H200. The smoke matrix has completed for the public and gated models that cleared harness validation: GPT-OSS 20B, Qwen 2.5 base, Qwen 2.5 Instruct, Qwen 3 8B, Llama 3.1 8B Instruct, Gemma 2 9B IT, Mistral 7B Instruct v0.3, OLMo 3 7B Instruct, and Phi-4. Chat-safety smoke outputs have been judged locally with Codex/Gemini where family separation permits, with Gemini-only judging for GPT-OSS-family outputs.

Completed powered runs with local audit judging currently include Qwen 2.5 7B Instruct, Llama 3.1 8B Instruct, and Gemma 2 9B IT. Llama audit judging is complete and pushed through commit `6768cb1`; Gemma artifacts and Codex/Gemini blinded-v3 judgments are complete and pushed through commit `7f5ca13`. The local clean development clone is pushed through commit `7efa5a4`, which keeps future local judging protocol-aware and defaults future judged rows to blinded-v3 with Codex gpt-5.4 plus Gemini.

The active H200 run is:

```text
selectivity_h200_powered_gpt_oss_20b
```

It is running in `/home/aryang9/sandbox/llm-safety/_worktrees/powered-gpt_oss_20b` at commit `276c091`. At the latest inspection on 2026-05-08, it was at 18.1% progress with the H200 GPU busy. Do not change code in that worktree and do not launch a duplicate. The run writes `generations.jsonl`, `cache_stats.jsonl`, and `progress.json` incrementally and is restartable with `--resume`.

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

Primary and diagnostic models:

| Key | Checkpoint | Track | Current status | Claim role |
| --- | --- | --- | --- | --- |
| `gpt_oss_20b` | `openai/gpt-oss-20b` | Chat-safety | Powered H200 run active now. | Primary cross-family model; GPT/OpenAI-family outputs use Gemini as the effective non-family judge. |
| `qwen2_5_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | Chat-safety | Powered run and local audit judging complete. | Primary Qwen instruction-tuned anchor. |
| `llama3_1_8b_instruct` | `meta-llama/Llama-3.1-8B-Instruct` | Chat-safety | Powered run and local audit judging complete. | Primary independent family replication. |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | Chat-safety | Powered run and local audit judging complete. | Primary independent family replication. |
| `mistral_7b_instruct_v0_3` | `mistralai/Mistral-7B-Instruct-v0.3` | Chat-safety | Queued after GPT-OSS completes. | Primary independent family replication. |
| `olmo3_7b_instruct` | `allenai/Olmo-3-7B-Instruct` | Chat-safety | Queued after Mistral unless decision gates say to stop. | Primary independent family replication. |
| `phi4` | `microsoft/phi-4` | Chat-safety | Queued; `microsoft/Phi-4-mini-instruct` remains the labeled fallback. | Primary independent family replication if operational. |
| `qwen3_5_9b` | `Qwen/Qwen3-8B` | Chat-safety | Registered text-only Qwen 3 replacement; queued after the broader-family panel. | Qwen architecture/version contrast, not an independent family. |
| `qwen2_5_7b_base` | `Qwen/Qwen2.5-7B` | Base-model | Registered diagnostic control; powered run queued after chat-family priority. | Alignment-contrast ablation for H6; not pooled with chat-refusal endpoints. |

The previously considered Qwen 3.5 multimodal line is stale for the active plan. The registered active candidate is `Qwen/Qwen3-8B`, a text-generation checkpoint that can run through the cache-intervention harness. The `qwen3_5_9b` config key is legacy naming only.

## Ablation And Safeguard Plan

In this plan, an ablation is a controlled comparison that isolates why an effect occurs. A safeguard is an intervention or model variant intended to preserve or recover safety behavior. These arms are diagnostic unless explicitly promoted before launch; they do not automatically count as independent cross-family replication.

| Arm | What it tests | Status |
| --- | --- | --- |
| Cache-policy ablations | Compare `none`, `sliding_window`, `sink_recent`, `random_matched`, `user_pinned`, and `policy_pinned` under the same prompt, model, seed, and retained-token budget. | Primary matrix for every powered model. |
| `policy_pinned` safeguard | Tests whether retaining system/policy-role cache tokens preserves safety more than generic retention. | Primary mitigation and causal-localization test. |
| `user_pinned` matched control | Tests whether the mitigation is policy-specific rather than just "keep more tokens." | Required control for H5. |
| Qwen 2.5 base-model ablation | Tests whether the selectivity pattern is tied to instruction tuning and safety alignment rather than generic cache sensitivity. | Registered as `qwen2_5_7b_base`; not a chat-refusal model unless a frozen scaffold is separately registered. |
| Simulated KV quantization | `kv_int8_sim` and `kv_int4_sim` test perturbation sensitivity, not eviction. | Phase 5 diagnostic only; report separately from eviction policies. |
| Provenance-pending safeguard/derivative models | `gpt-oss-sg`, `gpt-oss-der`, and `qwen3.5-unc` from `RESEARCH.md`. | Do not run or cite until exact source ID, revision/digest, owner, license, training method, base checkpoint, and runtime format are supplied. |

## Chloe Li / MSM Follow-Up Models

Use the official Model-Spec Midtraining code and artifacts first:

- Codebase: `https://github.com/chloeli-15/model_spec_midtraining`
- Spec folder: `https://github.com/chloeli-15/model_spec_midtraining/tree/main/spec/paper`
- Public Hugging Face collection/user: `https://huggingface.co/chloeli/collections`

Do not recreate an MSM model if an official public adapter already exists. Recreate only missing exact-panel variants, and only with the official procedure, hyperparameters, model-spec data path, and provenance logging.

Official adapter evaluation candidates:

| Candidate base | Official artifact role | Plan |
| --- | --- | --- |
| `Qwen/Qwen2.5-14B-Instruct` | Main official public MSM/AFT adapter scale that should fit for evaluation on H200. | First Chloe/MSM follow-up: evaluate the base model plus selected official adapters. |
| `Qwen/Qwen2.5-32B-Instruct` | Larger official public MSM/AFT adapter scale. | Evaluate only after GPU-only preflight confirms no CPU offload; do not train this size on the current setup. |
| `Qwen/Qwen3-14B` | Qwen 3 official MSM/AFT adapter scale. | Optional second official family/version follow-up after Qwen 2.5 results. |
| `Qwen/Qwen3-32B` | Larger Qwen 3 official MSM/AFT adapter scale. | Optional scale check if H200 memory and time allow. |

Adapter-stage ablations to prefer, when public for the selected base:

- Base model with no adapter.
- `msm` only.
- `aft-no-cot` or closest AFT-only public adapter.
- `msm-aft-no-cot`.
- `msm-aft-cot`, if available and budget allows.

Spec-content ablations to prefer:

- `rules-spec` and `rules-aug-spec` first, because they are closest to explicit safety-rule midtraining.
- Add `value-aug-spec`, `general-spec`, or `philosophy-spec` only if the first official MSM arm shows a meaningful contrast or if the paper needs breadth across spec types.

Exact-panel recreation candidates, only if official adapters are unavailable:

| Candidate | Why | Priority |
| --- | --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | Matched to our completed Qwen 7B powered run. | First recreation target if needed. |
| `Qwen/Qwen2.5-7B` | Matched base-model alignment contrast. | Optional; only if base-track MSM contrast is scientifically useful. |
| `Qwen/Qwen3-8B` | Active Qwen 3 text-only panel candidate. | Optional after the 7B instruct recreation. |
| `mistralai/Mistral-7B-Instruct-v0.3` or `allenai/Olmo-3-7B-Instruct` | Non-Qwen MSM generalization check. | Only one, only after primary panel results justify the extra training work and the official code can be used without undocumented changes. |

H200 hardware gate for Chloe/MSM work: the visible GPU is an NVIDIA H200 NVL with about 143 GiB VRAM, but the job cgroup has a 32 GiB host-RAM cap and no swap. Evaluate 14B and maybe 32B official adapters only when they can stay fully or almost fully on GPU. Avoid CPU offload. Faithfully recreating 14B or 32B MSM training is not planned on this single-H200 setup; if recreation is needed, prioritize 7B/8B exact-panel adapters.

## Interventions

Primary eviction/retention matrix:

- `none`
- `sliding_window`
- `sink_recent`
- `random_matched`
- `policy_pinned`
- `user_pinned`

`policy_pinned` is the cache-level safeguard: it protects system/policy tokens. `user_pinned` protects user tokens at the same retained-token budget and is the matched non-policy control needed for H5. Model-level safeguards, including Chloe Li / MSM adapters, are separate follow-up arms and should be analyzed against their own matching base checkpoints.

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
SELECTIVITY_STAGE=powered SELECTIVITY_MODELS=<model_key> \
  bash scripts/run_h200_selectivity_panel.sh
```

6. Watch progress:

```bash
uv run python scripts/report_selectivity_status.py \
  --run-dir results/selectivity_h200_powered_<model_key>
```

7. Commit and push run artifacts from H200:

```bash
bash scripts/h200_commit_run_artifacts.sh \
  results/selectivity_h200_powered_<model_key> master
```

8. Fetch to Mac. If the Mac worktree is too dirty to pull safely, use the tar fetch helper:

```bash
bash scripts/fetch_h200_selectivity_results.sh \
  results/selectivity_h200_powered_<model_key>
```

If the run lives in a dedicated H200 worktree, override the remote root:

```bash
H200_WORKDIR=/home/aryang9/sandbox/llm-safety/_worktrees/powered-<model_key> \
  bash scripts/fetch_h200_selectivity_results.sh \
  results/selectivity_h200_powered_<model_key>
```

For a best-effort partial snapshot while a run is still writing files, add `FETCH_PARTIAL=1`; complete run fetches should leave it unset.

## Local Judging Workflow

Judging happens on the Mac, not H200.

Primary judge command:

```bash
codex exec --cd /Users/aryan/Desktop/projects/llm-safety --sandbox read-only --ephemeral --output-last-message <tmpfile> -
```

Fallback judge command:

```bash
gemini --approval-mode plan --sandbox --output-format text -p '<prompt>'
```

For powered runs, judge sampled audit rows rather than blindly judging every generation row. The preferred completed-run handoff is:

```bash
uv run python scripts/sync_and_judge_selectivity_run.py \
  --run-id <run_id> \
  --workers 4
```

This fetches the H200-owned run directory, exports a per-model audit sample, approves that key JSONL for local model-judge data egress, and writes Codex/Gemini judgments under `docs/audit/`. Judge state is deliberately kept out of `results/<run_id>/`, because repeated H200 result syncs treat that directory as remote-owned.

For merged panel artifacts, the audit exporter preserves `source_run_id` and `model_id` scope so identical public prompt IDs from different models cannot collapse into one sampled item.

Manual equivalent:

```bash
uv run python scripts/export_human_audit_sample.py \
  --results-dir results/<run_id> \
  --output-dir docs/audit \
  --per-suite-policy 8 \
  --strategy effect \
  --include-hidden-reference \
  --seed 0

uv run python scripts/approve_judge_egress.py \
  --input-jsonl docs/audit/<run_id>_audit_key.jsonl \
  --output-jsonl docs/audit/<run_id>_audit_key.codex_gemini_approved.jsonl \
  --approval-note "User approved local Codex/Gemini judging for H200-generated selectivity audit rows." \
  --approval-source user_instruction \
  --overwrite

uv run python scripts/judge_with_codex_gemini.py \
  --input-jsonl docs/audit/<run_id>_audit_key.codex_gemini_approved.jsonl \
  --output-jsonl docs/audit/<run_id>_judgments.codex_gemini.jsonl \
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

- Monitor `selectivity_h200_powered_gpt_oss_20b` until completion or H200 expiry.
- If H200 expires, recover through the documented browser/JupyterHub token path, confirm no duplicate run is active, and resume the same run with `--resume`.
- When GPT-OSS completes, fetch, finalize, and push artifacts, then run Gemini-effective/family-safe local judging.
- Launch `mistral_7b_instruct_v0_3` on H200 while GPT-OSS judging runs on the Mac.
- Continue the queued primary panel in this order unless results or feasibility gates justify stopping: Mistral, OLMo, Phi or Phi-mini fallback, Qwen3-8B, then Qwen2.5-7B base.
- Use the completed-run `scripts/sync_and_judge_selectivity_run.py` handoff so local judge outputs cannot be overwritten by repeated H200 result syncs.
- Merge completed powered runs only after each per-model run has valid artifacts, readiness reports, audit coverage, and provenance manifests.
- After the primary cross-family panel is defensible, run the Chloe Li / MSM follow-up beginning with official Qwen2.5-14B adapters; evaluate 32B official adapters only after GPU-only memory preflight.
- Recreate exact-panel MSM adapters only when no official public adapter exists and only with the official `model_spec_midtraining` procedure.
- Keep provenance-pending derivative models (`gpt-oss-sg`, `gpt-oss-der`, `qwen3.5-unc`) out of the runnable plan until their source IDs, revisions, licenses, and training methods are documented.
- Extend paper analysis only from completed, provenance-valid runs; keep smoke, pilot, ablation, and external derivative evidence scoped accordingly.
