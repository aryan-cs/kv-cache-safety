# Selectivity Panel Findings Brief

Snapshot of cross-family selective-safety-erasure evidence as of
2026-05-14, sourced from the per-model `metrics.json` files and the
Claude-Sonnet-4.5-judged audit samples under `docs/audit/`.

## Headline Numbers

The top positive policy-level Selective Safety Erasure Index (SSEI) per
model, taken from `policy_level_contrasts` in each
`results/selectivity_h200_powered_<model>/metrics.json`:

| Family | Model | Top policy | SSEI [95% CI] |
| --- | --- | --- | --- |
| Qwen | Qwen3-9B | sliding window 128 | **0.092 [0.083, 0.101]** |
| Phi | Phi-4 | sliding window 128 | 0.084 [0.076, 0.091] |
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

- Six of eight instruction-tuned models (Qwen3-9B, Phi-4, OLMo-3,
  Llama-3.1, Qwen2.5-Instruct, Mistral) have at least one registered
  policy where positive SSEI's lower CI excludes zero. By the
  registered cross-family rule (>= 2 independent instruction-tuned
  families with positive selectivity), the claim **safety-minus-
  capability selectivity appears across multiple model families** is
  supported (Llama, OLMo, Phi, Qwen families clear the bar).
- Qwen3-9B is the panel maximum at 0.092 — the largest gap between
  safety degradation and capability degradation observed across the
  panel. Phi-4 is a close second at 0.084. Both effects come from the
  sliding_window cache eviction policy.
- Gemma-2 sits at zero — no selective safety erasure observed under any
  registered policy.
- GPT-OSS-20B has negative SSEI on every policy — capability degrades
  more than safety, the opposite pattern.

## Judging Provenance

All audit-row labels above come from Claude Sonnet 4.5 (`claude-sonnet-4-5`)
via the local `claude -p` CLI. Earlier Gemini-based labels for the same
audit samples are preserved alongside under `_judgments.gemini.jsonl`;
the Claude labels in `_judgments.claude.jsonl` are the canonical
judging artifacts used by the paper's cross-model figures and tables.
A 15-row calibration comparing Claude-Haiku vs Gemini on Mistral rows
showed ~93% agreement on unsafe-compliance / leakage / over-refusal and
~73% on refusal-correct; Sonnet should be at least as well-calibrated
as Haiku.

Coverage (parsed / total audit attempts) per model:

| Model | Parsed | Total | Parse rate |
| --- | --- | --- | --- |
| qwen2_5_7b_base | 44 | 44 | 100% |
| phi4 | 357 | 361 | 99% |
| gpt_oss_20b | 263 | 271 | 97% |
| qwen3_5_9b | 355 | 365 | 97% |
| qwen2_5_7b_instruct | 328 | 340 | 96% |
| gemma2_9b_it | 296 | 315 | 94% |
| llama3_1_8b_instruct | 321 | 349 | 92% |
| mistral_7b_instruct_v0_3 | 319 | 347 | 92% |
| olmo3_7b_instruct | 312 | 344 | 91% |

Overall: 2595 / 2676 parsed (97%).

## Caveats

1. **Phase 4 causal diagnostics have not been executed.** Without policy
   restoration vs matched user-role patching, the panel can only claim
   behavioral selectivity, not cache-mediated mechanism.
2. **Some small registered suites are under-powered.** The H200 panel
   shipped 1300 prompts/cell for the `public_*` suites but only 2-3
   prompts/cell for the registered `system_leakage`,
   `adversarial_refusal_safety`, and `base_alignment_contrast` suites.
   `scripts/check_publication_readiness.py` correctly flags these.

## File Map for the Paper

| Artifact | Purpose |
| --- | --- |
| `docs/generated/active_primary/family_replication_table.tex` | LaTeX `\maybeinputtable` target at `docs/latex/main.tex:358` |
| `docs/generated/cross_model_summary/cross_model_summary.tex` | Panel headline table (compact, 9 models) |
| `docs/generated/claim_assessment/abstract_status_sentence.tex` | Redefines `\EmpiricalStatusSentence` |
| `docs/generated/claim_assessment/claim_assessment_table.tex` | Per-claim verdict table |
| `docs/generated/cross_model_visuals/` | 15-figure exploratory gallery (`index.html` for one-page view) |
| `results/selectivity_h200_powered_<model>/figures/` | Per-model paper figures |

To rebuild all of the above after new data lands:

```bash
uv run python scripts/make_family_replication_table.py
uv run python scripts/make_cross_model_summary.py --provider claude
uv run python scripts/make_selectivity_claim_assessment.py
uv run python scripts/make_cross_model_visuals.py --provider claude
```

To rebuild per-model figures + tables, loop over each
`results/selectivity_h200_powered_<model>` and run
`scripts/make_figures.py` and `scripts/export_paper_assets.py`.
