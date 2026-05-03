# Human Audit Samples

Use `scripts/export_human_audit_sample.py` after a real run to create a stratified audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The generated CSV/JSONL files are local paper artifacts. Do not treat automated refusal-string scores as final unsafe-compliance labels without a small blinded human audit or a documented open local judge.
