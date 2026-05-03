# Human Audit Samples

Use `scripts/export_human_audit_sample.py` after a real run to create a stratified audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The script writes a blinded CSV for annotation and a private key JSONL that maps `audit_id` back to model, suite, policy, prompt id, seed, and automated metrics. Do not treat automated refusal-string scores as final unsafe-compliance labels without a small blinded human audit or a documented open local judge.

After annotation, aggregate the completed sheet:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/<run_id>_audit_blinded.csv \
  --key-jsonl paper/audit/<run_id>_audit_key.jsonl \
  --output-dir paper/audit/<run_id>_summary
```

The completed CSV may include an optional `annotator_id` column. Multiple rows with the same `audit_id` are treated as multiple annotations and are used to compute pairwise agreement and Cohen's kappa for each boolean label.

Accepted label values are `yes`, `no`, `true`, `false`, `1`, `0`, or blank. The aggregation writes:

- `human_audit_metrics.json`
- `human_audit_joined.csv`
- `human_audit_summary.md`
