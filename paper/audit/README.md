# Human Audit Samples

Use `scripts/export_human_audit_sample.py` after a real run to create a stratified audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The script writes a blinded CSV for annotation and a private key JSONL that maps `audit_id` back to model, suite, policy, prompt id, seed, and automated metrics. Do not treat automated refusal-string scores as final unsafe-compliance labels without a small blinded human audit or a documented open local judge.
