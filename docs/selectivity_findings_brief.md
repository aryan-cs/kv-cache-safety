# Selectivity Panel Findings Brief

Snapshot of cross-family selective-safety-erasure evidence as of
2026-05-14, sourced from the per-model `metrics.json` files and the
Gemini-judged audit samples under `docs/audit/`.

## Headline Numbers

The top positive policy-level Selective Safety Erasure Index (SSEI) per
model, taken from `policy_level_contrasts` in each
`results/selectivity_h200_powered_<model>/metrics.json`:

| Family | Model | Top policy | SSEI [95% CI] |
| --- | --- | --- | --- |
| Phi | Phi-4 | sliding window 128 | **0.084 [0.076, 0.091]** |
| OLMo | OLMo-3-7B-Instruct | user-pinned | 0.026 [0.019, 0.033] |
| Llama | Llama-3.1-8B-Instruct | user-pinned | 0.024 [0.016, 0.031] |
| Qwen | Qwen2.5-7B-Instruct | user-pinned | 0.017 [0.011, 0.022] |
| Mistral | Mistral-7B-Instruct-v0.3 | sliding window 128 | 0.009 [0.002, 0.017] |
| Gemma | Gemma-2-9B-IT | random matched | -0.000 [-0.003, 0.003] |
| OpenAI | GPT-OSS-20B | sink+recent | -0.068 [-0.088, -0.048] |

Qwen2.5-7B base is excluded (capability-only track; no safety contrasts).

## What the Numbers Mean

`SSEI = safety_degradation - capability_degradation`, both measured as
the change in pass rate moving from the no-cache-policy baseline to a
treatment cache policy, averaged over registered suites. Positive SSEI
means cache pressure damages safety behavior more than it damages
ordinary task capability. A CI excluding zero is the registered
significance criterion.

## Reading the Panel

- Five of seven instruction-tuned models (Phi-4, OLMo-3, Llama-3.1,
  Qwen2.5-Instruct, Mistral) have at least one registered policy where
  positive SSEI's lower CI excludes zero. By the registered
  cross-family rule (>= 2 independent instruction-tuned families with
  positive selectivity), the claim **safety-minus-capability selectivity
  appears across multiple model families** is supported by the current
  artifacts.
- Phi-4 is the panel maximum, with sliding-window cache eviction
  producing the largest selectivity gap.
- Gemma-2 and GPT-OSS-20B show flat or negative SSEI under every
  registered policy — these are the panel counterexamples.

## Caveats

1. **Judging coverage is uneven.** Gemini judging hit its daily quota
   mid-batch on 2026-05-13. Phi-4 (0/361), Qwen2.5-Instruct (0/340), and
   OLMo-3 (10/344) have effectively no current Gemini judgments and need
   to be rerun once the quota resets. The metrics for those models come
   from automated suite scoring, not from Gemini-judged audit rows; the
   numbers above are still valid but the audit-source disclosure must
   note the gap.
2. **Phase 4 causal diagnostics have not been executed.** Without policy
   restoration vs matched user-role patching, the panel can only claim
   behavioral selectivity, not cache-mediated mechanism.
3. **Some small registered suites are under-powered.** The H200 panel
   shipped 1300 prompts/cell for the `public_*` suites but only 2-3
   prompts/cell for the registered `system_leakage`,
   `adversarial_refusal_safety`, and `base_alignment_contrast` suites.
   `scripts/check_publication_readiness.py` correctly flags these.
4. **qwen3_5_9b is not yet pushed** from H200. Once it lands, rerun
   `scripts/make_family_replication_table.py`,
   `scripts/make_cross_model_summary.py`, and
   `scripts/make_selectivity_claim_assessment.py` to refresh the paper
   artifacts.

## File Map for the Paper

| Artifact | Purpose |
| --- | --- |
| `docs/generated/active_primary/family_replication_table.tex` | LaTeX `\maybeinputtable` target at `docs/latex/main.tex:358` |
| `docs/generated/cross_model_summary/cross_model_summary.tex` | Panel headline table (compact) |
| `docs/generated/claim_assessment/abstract_status_sentence.tex` | Redefines `\EmpiricalStatusSentence` |
| `docs/generated/claim_assessment/claim_assessment_table.tex` | Per-claim verdict table |
| `docs/generated/cross_model_visuals/` | 15-figure exploratory gallery |
| `results/selectivity_h200_powered_<model>/figures/` | Per-model paper figures |

To rebuild all of the above after new data lands:

```bash
uv run python scripts/make_family_replication_table.py
uv run python scripts/make_cross_model_summary.py
uv run python scripts/make_selectivity_claim_assessment.py
uv run python scripts/make_cross_model_visuals.py
```

To rebuild per-model figures + tables, loop over each
`results/selectivity_h200_powered_<model>` and run
`scripts/make_figures.py` and `scripts/export_paper_assets.py`.
