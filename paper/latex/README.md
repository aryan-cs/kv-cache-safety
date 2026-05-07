# LaTeX Manuscript

This directory contains the arXiv-friendly paper draft for Cache-Mediated Safety Erasure.

The current default is a plain `article` preprint instead of a conference-specific style, because the target venue is not fixed and arXiv does not require a special template. Once a target venue is selected, this source can be adapted to that venue's official style.

Build the readable PDF from the repository root:

```bash
bash scripts/build_paper_pdf.sh
```

The output PDFs are written to:

```text
docs/build/kv-cache-safety.pdf
docs/kv-cache-safety.pdf
```

The manuscript is intentionally marked as a pre-results draft. Do not submit it until the H200 sweeps, causal patch runs, human audit, figures, table manifests, and publication readiness checks have passed on a clean commit.
