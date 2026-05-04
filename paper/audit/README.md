# Human Audit Samples

Use `scripts/export_human_audit_sample.py` after a real run to create a stratified audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The script writes a blinded CSV for annotation and a private key JSONL that maps `audit_id` back to model, suite, policy, prompt id, seed, hidden/system text, and automated metrics. By default, the blinded CSV redacts raw system and hidden-system text and includes only short digests. Use `--include-hidden-reference` only for a leakage-specific audit where annotators must compare the model response against the hidden reference. Do not treat automated refusal-string scores as final unsafe-compliance labels without a small blinded human audit or a documented open local judge.

Use [labeling_guide.md](labeling_guide.md) when completing the blinded CSV. The export samples prompt-matched baseline/treatment pairs so the aggregation can compute paired human-audit deltas. By default, the exporter prioritizes pairs with the largest automated safety, leakage, or over-refusal shifts so human effort concentrates on claim-relevant examples. Add `--strategy random` for unbiased spot checks.

To create duplicate blinded sheets for inter-annotator agreement, add:

```bash
uv run python scripts/export_human_audit_sample.py \
  --results-dir results/<run_id> \
  --annotator-template-count 2
```

This writes the standard blinded CSV plus `*_annotator_01.csv`, `*_annotator_02.csv`, and so on with prefilled `annotator_id` values. Aggregate the completed annotator files together with one `--audit-csv` argument per file.

After annotation, aggregate the completed sheet:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/<run_id>_audit_blinded.csv \
  --key-jsonl paper/audit/<run_id>_audit_key.jsonl \
  --output-dir paper/audit/<run_id>_summary
```

For the publication build, the expected summary directories are:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/h200_qwen_full_sweep_audit_blinded.csv \
  --key-jsonl paper/audit/h200_qwen_full_sweep_audit_key.jsonl \
  --results-dir results/h200_qwen_full_sweep \
  --output-dir paper/audit/h200_qwen_full_sweep_summary

uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/h200_causal_patch_qwen7b_audit_blinded.csv \
  --key-jsonl paper/audit/h200_causal_patch_qwen7b_audit_key.jsonl \
  --results-dir results/h200_causal_patch_qwen7b \
  --output-dir paper/audit/h200_causal_patch_qwen7b_summary
```

The completed CSV may include an optional `annotator_id` column. Multiple rows with the same `audit_id` are treated as multiple annotations only when they come from distinct annotator IDs. Duplicate `(audit_id, annotator_id)` rows are deduplicated, reported in the summary, and block publication readiness.

Accepted label values are `yes`, `no`, `true`, `false`, `1`, `0`, or blank. The aggregation writes:

- `human_audit_metrics.json`
- `human_audit_summary.json`
- `human_labels.jsonl`
- `human_audit_joined.csv`
- `human_audit_summary.md`
- `human_audit_summary_table.tex`
- `human_audit_deltas_table.tex`
- `audit_manifest.json`

The JSON summary reports publication-facing label rates at the item level after majority consensus across annotators; unresolved ties are listed and block readiness. It also keeps annotation-level label rates as diagnostics, includes Wilson confidence intervals, automated-vs-human confusion matrices, pairwise inter-annotator agreement across distinct annotators, duplicate-annotation diagnostics, and paired baseline-vs-policy deltas when the same `prompt_id` and `seed` appear under `none` and a treatment policy.

Before using the audit in the paper, run:

```bash
uv run python scripts/check_human_audit_readiness.py \
  --summary-json paper/audit/<run_id>_summary/human_audit_summary.json \
  --require-baseline-deltas
```

By default this requires complete annotations, no unknown audit IDs, no duplicate `(audit_id, annotator_id)` rows, at least two distinct annotators, non-empty core safety labels, paired treatment-minus-baseline deltas, and at least one inter-annotator pair for each core label. Use `--allow-single-annotator` only for a clearly documented draft or ablation.
