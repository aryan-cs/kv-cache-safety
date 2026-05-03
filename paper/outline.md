# Paper Outline: Cache-Mediated Safety Erasure

## Working Thesis

Safety/refusal behavior in aligned LLMs is partly implemented through fragile cache-resident routing state. Common inference-time KV-cache optimizations can erase this state, causing safety failures that are not predicted by ordinary capability degradation.

## Target Contributions

1. Demonstrate selective safety degradation under KV-cache eviction/quantization at compression levels where benign capability remains comparatively stable.
2. Localize degradation by token span, layer, and head where possible.
3. Show causal restoration: patching or preserving specific cache slices restores refusal behavior.
4. Propose and test `policy_pinned` cache retention as an alignment-preserving mitigation.

## Key Figures

1. Safety degradation vs capability degradation across cache policies and budgets.
2. Selective Safety Erasure Index by model and policy.
3. Token-span retention heatmaps for system/policy/user spans.
4. Causal patching restoration curves by layer/head/span.
5. Mitigation comparison: naive compression vs policy-pinned compression.

## Acceptance Criteria For A Paper Claim

- Effects reproduce on at least two open model families or two sizes within one family.
- Safety/refusal degradation is meaningfully larger than ordinary capability degradation.
- At least one causal patching experiment restores a safety metric without globally restoring the full cache.
- All runs are reproducible from committed configs and scripts.
