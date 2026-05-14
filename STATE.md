# Current Project State

This file is a handoff summary for a new chat. It captures the important context from the long monitoring/design thread for `/Users/aryan/Desktop/projects/llm-safety`.

## Selectivity Panel Status (2026-05-14)

- Eight `selectivity_h200_powered_*` runs have completed locally on the Mac:
  qwen2_5_7b_base, qwen2_5_7b_instruct, gpt_oss_20b, llama3_1_8b_instruct,
  gemma2_9b_it, mistral_7b_instruct_v0_3, olmo3_7b_instruct, phi4.
- A ninth model, `qwen3_5_9b`, is being tested on H200; its run has not yet been
  pushed to `origin/master` and the Mac does not yet have results for it.
- Codex has been completely removed from the repository (commit `c86c5ff` and
  follow-ups). The only supported local judge is Gemini via
  `scripts/judge_with_gemini.py`.
- Gemini judging coverage by model (parsed / attempts):

  | Model | Parsed | Attempts | Status |
  | --- | --- | --- | --- |
  | qwen2_5_7b_base | 44 | 44 | complete |
  | gpt_oss_20b | 270 | 271 | complete |
  | mistral_7b_instruct_v0_3 | 344 | 347 | complete |
  | gemma2_9b_it | 315 | 315 | complete |
  | llama3_1_8b_instruct | 460 | 460 | complete |
  | olmo3_7b_instruct | 10 | 344 | partial; quota blocked |
  | phi4 | 0 | 361 | quota blocked |
  | qwen2_5_7b_instruct | 0 | 340 | quota blocked |

  The blocked models hit Gemini's daily quota mid-batch on 2026-05-13. Retry
  after the quota resets (~20h from the block) with:

  ```bash
  uv run python scripts/judge_with_gemini.py \
    --input-jsonl docs/audit/selectivity_h200_powered_<model>_audit_key.gemini_approved.jsonl \
    --output-jsonl docs/audit/selectivity_h200_powered_<model>_judgments.gemini.jsonl \
    --providers gemini --workers 4 --resume --allow-data-egress \
    --retry-statuses blocked,parse_error,unlabeled
  ```

- Selectivity claim-ladder verdicts (`docs/generated/claim_assessment/`):

  | Claim | Status |
  | --- | --- |
  | behavioral_cache_sensitivity | supported |
  | safety_minus_capability_selectivity | supported |
  | cross_family_replication | supported (Llama, OLMo, Phi, Qwen) |
  | audit_provenance_complete | supported (5/8 ≥95% Gemini coverage) |
  | targeted_mitigation | pending (Phase 4 not run) |
  | causal_localization | pending (Phase 4 not run) |
  | alignment_contrast | pending (qwen base only 2 prompts/cell) |

  Publication-ready: false (3 claims pending external work).

- Paper artifacts written to `docs/generated/`:
  - `docs/generated/selectivity_h200_powered_<model>/` — per-model tables + figures
  - `docs/generated/active_primary/` — Mistral promoted as manuscript anchor
  - `docs/generated/active_primary/family_replication_table.{md,tex}`
  - `docs/generated/cross_model_summary/cross_model_summary.{json,md,tex}`
  - `docs/generated/cross_model_visuals/` — 15-figure gallery
  - `docs/generated/claim_assessment/` — `\EmpiricalStatusSentence` redefinition + claim table

- Publication readiness blockers (per `scripts/check_publication_readiness.py`):
  - registered `system_leakage` and `adversarial_refusal_safety` suites only have 2–3 prompts per cell across all 8 runs; the H200 panel was powered for the `public_*` variants but not these small registered confirmatory suites
  - Phase 4 causal diagnostics not yet executed for any model
  - matched base-model alignment-contrast track has only 2 prompts/cell for qwen2_5_7b_base

  These are data-collection gaps that require H200 reruns; they are not
  fixable from the Mac side.

## High-Level Status

- Overall paper progress is about **96%** for the old Qwen-centered evidence-gated draft.
- The old pipeline is **not publication-ready** as a strong positive causal-erasure paper.
- The current Qwen results are useful pilot evidence, but the study needs to be redesigned as a multi-model, cross-family robustness study.
- A new research plan was started in `RESEARCH.md`.

## Main Scientific Takeaway So Far

The current data suggests:

- KV-cache interventions can change model behavior.
- Safety-relevant behavior appears to be affected in some cases.
- The evidence for selective safety degradation is mixed.
- The strongest causal claim, that a specific cache-resident safety state was found and restored, is not cleanly supported.
- Causal confidence intervals remain too wide for strong H3-style claims.
- The work should be reframed as a cautious, evidence-gated study unless new cross-family experiments support stronger claims.

In simple terms: **the current data says cache changes can perturb safety behavior, but it does not yet prove a general “KV-cache safety erasure” mechanism.**

## Completed H200 Runs

Known completed evidence:

- Primary CI extension:
  - run id: `h200_qwen14b_ci_extension_primary`
  - status: complete
  - progress: `23418 / 23418`
  - cache stats: valid `cache_stats.parquet`
  - important note: older corrupt partial artifacts were archived and must not be restored.

- Merged primary evidence:
  - run id: `h200_qwen_full_sweep_plus_ci_extension`
  - status: present on H200
  - rows observed earlier: `48699`
  - used for primary open-local-judge audit.

- Causal CI extension:
  - run id: `h200_causal_patch_qwen7b_ci_extension`
  - status: complete
  - progress: `9114 / 9114`
  - synced locally to `results/h200_causal_patch_qwen7b_ci_extension`
  - strict causal readiness still fails due CI-width gates.

## Latest H200 Connectivity State

- H200 was temporarily unreachable through the Jupyter SSH proxy with `403 Forbidden` / `302 Found`.
- Aryan said it should be back up.
- A subsequent check succeeded:
  - H200 reachable
  - H200 checkout clean
  - H200 commit: `474d1fb`
  - branch: `master`
  - ahead/behind: `0 / 0`
  - no active experiment/audit processes at that moment
  - GPU showed memory in use but no active tracked process in the audit/experiment grep.

If monitoring resumes, first inspect H200 again rather than assuming this is still current.

## Mac Repo State

Local repo path:

`/Users/aryan/Desktop/projects/llm-safety`

Known local state:

- branch: `master`
- local head: `4545384`
- cached `origin/master`: `474d1fb`
- ahead/behind: `0 / 19`
- dirty count: about `55`
- Mac `git fetch --prune origin` repeatedly failed with:
  - `error: cannot open '.git/FETCH_HEAD': Operation not permitted`
- Do not run `git pull` on the Mac while it is dirty.
- Do not reset, checkout, or rebase to force sync.

The worktree contains many pre-existing changes plus current changes. Be careful not to revert user work.

## Important Local Artifacts

The paper was moved from `paper/` to `docs/` for GitHub Pages hosting.

Important local paths:

- PDF: `docs/kv-cache-safety.pdf`
- LaTeX source: `docs/latex/main.tex`
- generated primary assets: `docs/generated/...`
- audit assets: `docs/audit/...`
- current causal CI local results: `results/h200_causal_patch_qwen7b_ci_extension`
- new research plan: `RESEARCH.md`
- this handoff: `STATE.md`

The PDF was previously rebuilt and visually checked after layout fixes. Page 16/page 17 overlap issues were addressed by simplifying the dense suite-effects table.

## Open-Local-Judge Audit State

Aryan asked for AI/manual auditing, but the implementation must not call AI labels human labels.

Open-local-judge audit evidence is diagnostic only unless the paper clearly says so.

Completed/synced audit artifacts:

- Primary audit input:
  - `docs/audit/h200_qwen_full_sweep_plus_ci_extension_audit_blinded.csv`
  - rows: `672`
- Primary open-judge variants:
  - `v1`: `672 / 672`
  - `v2`: `672 / 672`
  - `v3`: `672 / 672`
- Causal audit input:
  - `docs/audit/h200_causal_patch_qwen7b_ci_extension_audit_blinded.csv`
  - rows: `199`
- Causal open-judge variants:
  - `v1`: `199 / 199`
  - `v2`: `199 / 199`
  - `v3`: `199 / 199`

Total open-local-judge variant rows completed:

- `2613 / 2613`

Aggregation status:

- H200 aggregation wrote:
  - `paper/audit/h200_qwen_full_sweep_plus_ci_extension_summary`
- It then failed readiness with:
  - completion rate below `1.0`
  - missing raw-output hashes for some annotation rows
  - some unlabeled rows
  - several unresolved consensus ties
  - multi-annotator fractions below `1.0`

Important: this means the audit files exist, but the audit-readiness gate does **not** pass.

## Known Audit Problem

The open judge was run with `--on-parse-error record_unlabeled`, so malformed judge outputs were preserved as unlabeled rows. This is honest, but it blocks readiness.

Likely next audit work:

1. Inspect unlabeled rows in the open-judge CSVs.
2. Determine whether parse failures were due prompt formatting, judge verbosity, JSON extraction weakness, or true ambiguity.
3. Improve the judge prompt/parser if needed.
4. Rerun only missing/unlabeled audit rows, not the full audit, if the scripts support it.
5. Keep raw-output hashes for every row.
6. Keep source type as `open_local_judge`, not human.

Do not synthesize labels into human annotator templates.

## Current Blockers

For the old Qwen-centered paper:

1. Causal CI-width gates still fail.
2. Open-local-judge audit readiness fails.
3. Real human audit labels are absent/incomplete.
4. Strong H2/H3 claims are not supported.
5. The study is too Qwen-heavy for model-family claims.

For the redesigned project:

1. Need model configs for non-Qwen families.
2. Need a cross-family runner/launcher.
3. Need a judge wrapper that prevents same-family judging.
4. Need revised readiness gates for cross-family claims.
5. Need staged H200 runs for smoke, powered behavioral sweeps, audits, and causal diagnostics.

## Research Redesign Decision

Aryan said the experiment should be reworked because testing only one model/family is not enough and the judge should be different from the evaluated model.

Agreed direction:

- Treat existing Qwen results as pilot evidence.
- Redesign the study as cross-family.
- Test more than one model type/family:
  - Qwen
  - Llama
  - Gemma
  - GPT-OSS if feasible
  - optional Mistral/Phi style model if compute allows
- Use cross-family judges:
  - Do not judge Qwen with Qwen.
  - Prefer GPT-OSS as judge if available and stable.
  - Otherwise use a separate family such as Llama or Gemma as judge.
- Make claim strength depend on replication:
  - no non-Qwen result means no broad model-family claim.

See `RESEARCH.md` for the new proposed protocol.

## Files Changed In This Thread

Important files added/changed:

- `RESEARCH.md`
  - new cross-family research plan
- `STATE.md`
  - this handoff
- `docs/latex/main.tex`
  - scoped claims, limitations, audit language, causal/quantization/attention wording
- `README.md`
  - added evidence scope, missing validation, audit caveats
- `docs/manuscript.md`
  - clarified audit source language
- `docs/experiment_log.md`
  - updated status snapshot
- `scripts/aggregate_human_audit.py`
  - added response length calibration fields/summary
- `scripts/check_human_audit_readiness.py`
  - open-judge readiness now requires calibration/provenance
- `scripts/export_paper_assets.py`
  - simplified cramped suite-effects table
- tests updated for audit/readiness/status behavior

Also:

- paper folder was moved to `docs/`
- PDF renamed to `docs/kv-cache-safety.pdf`
- generated/audit artifacts were synced from H200 to `docs/...`

## Verification Already Run

Previously passed:

- `uv run ruff check .`
- `bash -n scripts/*.sh`
- targeted tests for audit/readiness/status/LaTeX
- broader tests around artifacts, audit, publication status, prepared suites, prompt disjointness
- PDF manifest validation
- visual PDF page checks after table/layout fixes

Known expected failure:

- `check_latex_placeholders.py` failed because real causal audit summary artifacts were missing at that time.
- Do not fix that by fabricating labels.

## What To Do Next

Recommended next steps in a new chat:

1. Read `RESEARCH.md`.
2. Decide the initial cross-family model set based on what H200 can load:
   - Qwen existing artifact
   - Llama 8B-class instruct
   - Gemma 9B-class instruct
   - GPT-OSS if available
3. Add model config YAMLs for the selected non-Qwen models.
4. Add a small cross-family smoke config or launcher.
5. Add a judge-family guard so evaluated model family and judge family cannot match by accident.
6. Fix open-judge parser/provenance issues for diagnostic audits.
7. Rebuild the paper around the new cross-family plan.
8. Do not run 32B until the cross-family plan is explicit and smaller models are stable.

## Operating Rules To Preserve

- Always report both:
  - overall paper progress
  - current blocking task progress
- Do not conflate paper progress with active run progress.
- Never resume `h200_*` run ids on the Mac.
- Do not fabricate labels, fake data, or filler data.
- Do not call AI/open-judge labels human labels.
- Do not overwrite dirty Mac source files.
- Do not pull the Mac repo while dirty.
- Do not run duplicate H200 launchers while a lock/process exists.
- H200 work must happen in:
  - `/home/aryang9/sandbox/llm-safety`
- Mac work happens in:
  - `/Users/aryan/Desktop/projects/llm-safety`

## Plain-English Summary

We have a real Qwen pilot showing cache interventions can affect safety-relevant behavior, but it is not strong enough for a broad publication claim. The next version of the project should test whether this behavior appears across multiple model families and use judges from different families. The project is shifting from “prove KV-cache safety erasure in Qwen” to “measure when and where KV-cache changes perturb safety across model families, and which mitigations help.”
