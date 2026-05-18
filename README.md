# The Safety Tax of Cache Compression

**Paper:** [docs/kv-cache-safety.pdf](docs/kv-cache-safety.pdf) | **LaTeX source:** [docs/latex/main.tex](docs/latex/main.tex) | **Repository:** [github.com/aryan-cs/kv-cache-safety](https://github.com/aryan-cs/kv-cache-safety)

Production LLM serving systems compress or evict the key-value (KV) cache to reduce memory cost. We show that this throughput optimization selectively weakens refusal and policy-following behavior more than ordinary task capability on dense full-attention models. We measure this effect across twelve open-weight checkpoints from seven model families and provide causal-patching evidence that cache state is a safety-relevant surface. A simple mitigation (policy-pinned retention of system-role tokens) fully restores refusal behavior across all tested architectures.

NeurIPS 2025 preprint.

## Key Results

- **Selective safety erasure is real and widespread.** Eight of twelve instruction-tuned models show positive SSEI (safety degradation minus capability degradation) with 95% bootstrap CIs excluding zero under sliding-window or user-pinned cache policies.
- **Four model families affected.** Llama, OLMo, Phi, and Qwen all exhibit the effect (SSEI 0.02 to 0.09 in absolute pass-rate units). The result survives leave-one-family-out perturbation.
- **Three models are immune.** Gemma-2-9B-IT, Mistral-7B-Instruct-v0.3, and gpt-oss-20b do not show the pattern. All three have architectural features (interleaved sliding-window attention, full-layer sliding-window, MoE with harmony formatting) that decouple long-range cache state from the safety circuit.
- **Alignment reduces but does not eliminate the effect.** A Qwen2.5-7B base model shows SSEI = 0.162 vs. 0.017 for its instruction-tuned counterpart.
- **Causal patching confirms the mechanism.** Restoring baseline K+V into compressed runs recovers 22-58% of lost refusal across Qwen, Llama, and Phi. System-role and user-role restorations produce comparable recovery, indicating safety information is distributed across cached tokens.
- **Policy-pinned retention fully mitigates the effect.** Protecting system-role tokens from eviction restores refusal behavior across all tested models.

## Repository Layout

```
src/cache_safety_erasure/        Core Python package
  cache_policies/                KV-cache compression policy implementations
  generation/                    Generation runner, HF generation loop, causal patching
  analysis/                      Result aggregation and analysis
  evals/                         Evaluation harnesses
  judging/                       Safety judge implementations
  metrics/                       SSEI and per-suite metric computation
  models/                        Model loading and device-map config
  utils/                         Shared utilities

configs/
  experiments/                   Experiment YAML configs (H200 sweeps, causal patching, budget sweeps)
  models/                        Model panel definitions
  prompt_suites/                 Prompt suite configurations

scripts/                         Runnable scripts (experiments, analysis, paper build)
tests/                           Unit and integration tests (16 test modules)
docs/
  latex/                         Paper source (main.tex, neurips_2025.sty)
  audit/                         Blinded human-audit and judge-audit sheets
  generated/                     Auto-generated tables, figures, and claim assessments
results/                         Experiment output directories (one per run)
```

## Setup

Requires Python >= 3.11. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/aryan-cs/kv-cache-safety.git
cd kv-cache-safety
uv sync --extra dev
```

Prepare the built-in diagnostic prompt suites:

```bash
uv run python scripts/prepare_data.py --suite all
```

Run the test suite:

```bash
uv run pytest
uv run ruff check .
```

## Running Experiments

Every experiment is defined by a YAML config in `configs/experiments/`. The main entry point is:

```bash
uv run python scripts/run_experiment.py --config configs/experiments/<config>.yaml
```

**Smoke tests** (no GPU required):

```bash
# Mock model (deterministic, fast)
uv run python scripts/run_experiment.py --config configs/experiments/smoke_mock.yaml

# Tiny HF model (checks generation plumbing)
uv run python scripts/run_experiment.py --config configs/experiments/tiny_hf_smoke.yaml
```

**Resume a partial run:**

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/<config>.yaml \
  --run-id <run_id> \
  --resume
```

Resume is fail-closed: it refuses to append if the model, prompt suites, policy matrix, seeds, expected row count, or git commit do not match the original run.

## Cache Policies

| Policy | Description |
|--------|-------------|
| `none` | Uncompressed baseline |
| `sliding_window` | Keep last N cached tokens |
| `sink_recent` | Keep first S (sink) plus last N (recent) cached tokens |
| `random_matched` | Random eviction matched to the same budget |
| `policy_pinned` | Protects system-role token spans from eviction (mitigation) |
| `user_pinned` | Protects user-role token spans from eviction (control) |
| `kv_int8_sim` | Symmetric per-tensor int8 quantize/dequantize simulation |
| `kv_int4_sim` | Symmetric per-tensor int4 quantize/dequantize simulation |
| `attention_h2o` | Diagnostic: sink/recent + high-attention historical tokens |

For causal diagnostics, `patch_from_baseline` restores baseline K+V tensors at specific token positions into compressed runs, targeting by `token_roles` (e.g., system, user).

## Analysis and Figures

After a run completes, generate aggregate metrics and figures:

```bash
uv run python scripts/aggregate_results.py --results-dir results/<run_id>
uv run python scripts/make_figures.py --results-dir results/<run_id>
```

Cross-model analysis scripts:

```bash
uv run python scripts/make_cross_model_summary.py       # Cross-family SSEI summary
uv run python scripts/make_cross_model_visuals.py        # Cross-family visualizations
uv run python scripts/make_family_replication_table.py   # Leave-one-out replication
uv run python scripts/make_robustness_analysis.py        # Robustness checks
uv run python scripts/make_selectivity_claim_assessment.py  # Claim-ladder gate checks
uv run python scripts/make_budget_dose_response.py       # Budget sweep dose-response
uv run python scripts/assess_claims.py                   # H1-H6 claim assessment
```

Export paper assets (LaTeX tables):

```bash
uv run python scripts/export_paper_assets.py --results-dir results/<run_id>
```

## Building the Paper

The manuscript is in `docs/latex/main.tex` (NeurIPS 2025 preprint format). Build with [tectonic](https://tectonic-typesetting.github.io/):

```bash
bash scripts/build_paper_pdf.sh
```

This produces `docs/latex/main.pdf` and copies it to `docs/kv-cache-safety.pdf`.

Package for arXiv submission:

```bash
uv run python scripts/package_arxiv_submission.py
```

## Auditing

The project uses blinded human audits and open-model judge audits for label validation. Audit tooling:

```bash
# Export blinded audit sample
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>

# Aggregate completed annotations
uv run python scripts/aggregate_human_audit.py

# Run open-model judge audit
uv run python scripts/run_open_judge_audit.py
```

Audit sheets live in `docs/audit/`. Judge labels are source-marked as `open_local_judge` and require model provenance and prompt-template hashes.

## Experiment Configs

The `configs/experiments/` directory contains all registered experiment configurations:

**Cross-family selectivity panel** (one per model):
- `selectivity_h200_powered_qwen2_5_7b_instruct.yaml`, `..._14b_instruct.yaml`, `..._7b_base.yaml`
- `selectivity_h200_powered_llama3_1_8b_instruct.yaml`
- `selectivity_h200_powered_phi4.yaml`
- `selectivity_h200_powered_olmo3_7b_instruct.yaml`
- `selectivity_h200_powered_mistral_7b_instruct_v0_3.yaml`
- `selectivity_h200_powered_gemma2_9b_it.yaml`
- `selectivity_h200_powered_gpt_oss_20b.yaml`
- `selectivity_h200_powered_qwen3_5_9b.yaml`

**Causal patching** (cross-family):
- `h200_causal_patch_qwen3_5_9b.yaml`
- `h200_causal_patch_llama3_1_8b_instruct.yaml`
- `h200_causal_patch_phi4.yaml`

**Budget dose-response sweeps**:
- `selectivity_h200_budget_sweep_llama3_1_8b_instruct.yaml`
- `selectivity_h200_budget_sweep_phi4.yaml`
- `selectivity_h200_budget_sweep_qwen2_5_14b_instruct.yaml`

**MSM (Model Spec Midtraining) variants**:
- `selectivity_h200_powered_qwen2_5_14b_msm_rules.yaml`
- `selectivity_h200_powered_qwen2_5_14b_msm_value_aug.yaml`

**Other**:
- `selectivity_h200_base_alignment_expanded.yaml` (base vs. instruct alignment contrast)

## Artifact Contract

Every experiment run writes to `results/<run_id>/`:

| File | Contents |
|------|----------|
| `config.resolved.yaml` | Fully resolved config |
| `environment.json` | Python, platform, package, device metadata |
| `manifest.json` | Run metadata, git commit, model config, policy matrix, expected row count |
| `prompts.jsonl` | Rendered prompts with hashes, token IDs, and token-role spans |
| `generations.jsonl` | Generated text and per-example metrics |
| `metrics.json` | Aggregate suite/policy metrics with SSEI and bootstrap CIs |
| `cache_stats.parquet` | Retained/evicted token counts by policy, layer, and role |
| `figures/` | PNG, SVG, PDF plots with source CSVs and manifest |

Generated paper assets go to `docs/generated/<run_id>/` (LaTeX tables, claim assessments).

## Hardware

- **Development:** MacBook with Apple Silicon (24 GB unified memory). Sufficient for tests, smoke runs, and paper builds.
- **Full experiments:** NVIDIA H200 (141 GB VRAM). Required for the 7B-14B model panel. Configs reject CPU/disk offload.

## SSEI Metric

The Selective Safety Erasure Index is defined as:

```
SSEI = safety_degradation - capability_degradation
```

where each degradation term is the drop in pass rate (baseline minus compressed) on safety or capability prompt suites respectively. Positive SSEI means the cache policy hurts safety more than capability. All reported CIs use prompt-clustered bootstrap resampling.

## Safety and Data Policy

This repository is for safety evaluation research. Built-in prompt suites avoid procedural harmful details. All judges are open local models (no closed-source endpoints or paid APIs). Judge labels require model provenance and prompt-template hashes before they can support claim assessment.

## Citation

```bibtex
@article{gupta2025safetytax,
  title={The Safety Tax of Cache Compression},
  author={Gupta, Aryan},
  year={2025},
  note={NeurIPS 2025 preprint}
}
```

## License

Research code. See [pyproject.toml](pyproject.toml) for package metadata.
