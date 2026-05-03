# Testing Cache-Mediated Safety Erasure in Open LLMs

> Status: preregistered draft scaffold. Do not submit until the H200 sweeps, causal patch runs, figures, and statistical checks have completed on a clean git commit. Mock and tiny-model runs validate plumbing only and are not paper evidence.

## Abstract

Modern LLM deployment stacks increasingly compress, evict, or quantize KV caches to improve inference throughput. Prior work shows that KV-cache compression can unevenly degrade instructions and increase system-prompt leakage. We test a narrower alignment hypothesis: refusal behavior may depend on cache-resident routing state that is more fragile than ordinary task capability. The project evaluates this hypothesis with open models, public prompts, prompt-clustered confidence intervals, token-role cache accounting, mitigation via policy-span pinning, and causal restoration experiments for length-preserving cache perturbations. We use the term **Cache-Mediated Safety Erasure** only if results show that targeted cache restoration or pinning recovers refusal more than matched non-policy controls.

## 1. Introduction

Safety evaluations usually treat a deployed model as a fixed pair of weights and prompts. Real serving systems are not fixed in this way: transient inference state is altered by KV-cache eviction, quantization, batching, paging, and memory-management policies. If refusal behavior depends on transient state, then serving optimizations can become safety interventions.

This paper tests a claims ladder:

1. **Behavioral cache sensitivity:** cache policies change safety, leakage, or capability behavior.
2. **Selective safety degradation:** safety behavior degrades more than ordinary capability under matched cache pressure.
3. **Causal safety-state erasure:** restoring or protecting specific policy-token cache slices restores safety more than matched irrelevant slices.

The third claim is the novel target. If only the first two claims hold, the paper should be framed as a replication and extension of KV-compression pitfall work rather than as a new safety-erasure phenomenon.

## 2. Related Work

KV-cache compression work shows why this question is plausible. H2O, StreamingLLM, SnapKV, KIVI, KVQuant, and MiKV propose cache retention or quantization methods for efficient inference. The closest behavioral overlap is **The Pitfalls of KV Cache Compression**, which reports uneven instruction degradation and system-prompt leakage under compression. The closest mechanistic overlap is **Understanding the Physics of Key-Value Cache Compression**, which frames compression as perturbing token-routing accessibility. CachePrune shows that editing KV-cache state can also be a security defense against indirect prompt injection.

The alignment mechanism motivation comes from refusal-direction and alignment-routing work: refusal can be steered by activation-space interventions, and recent work suggests policy behavior can route through sparse heads or directions. Our planned novelty is not that compression can hurt performance, nor that KV state matters. The planned novelty is a causal safety-specific result: policy/refusal cache state is selectively fragile and recoverable by targeted cache preservation or restoration.

Safety evaluation uses public refusal and robustness anchors such as HarmBench, JailbreakBench, XSTest, and IFEval. Long-context controls should use RULER and LongBench-style tasks where feasible.

## 3. Methods

### Models

Primary sweeps use open Qwen instruction models at 7B, 14B, and 32B scale. A paper-grade generalization claim requires at least one non-Qwen open model family, or the claim must be explicitly limited to Qwen-family models.

### Cache Interventions

We compare baseline decoding against:

- sliding-window eviction
- sink-plus-recent retention
- random matched eviction
- H2O-style attention retention on diagnostic subsets
- simulated int8/int4 K/V perturbation
- policy-pinned retention
- baseline-cache restoration for length-preserving perturbations

The current restoration engine patches retained cache positions. It can test quantization and retained-position restoration, but it cannot yet reinsert evicted tokens. Eviction claims therefore rely on policy-pinning and matched negative controls until reinsertion-style patching is implemented.

### Prompt Suites

The experiment separates safety and capability suites:

- `system_leakage`: attempts to reveal hidden system/policy text.
- `public_refusal_safety`: public harmful-request prompts prepared through `scripts/prepare_data.py`.
- `public_benign_overrefusal`: benign prompts that should not be refused.
- `public_capability_arc`: public multiple-choice capability controls.
- optional long-context suites: RULER or LongBench-derived subsets.

Every run records raw prompt fields, rendered chat text, tokenizer token IDs, exact token offsets when available, token-role spans, prompt hashes, dataset provenance, full cache policy configs, and environment state.

### Metrics

The main behavioral metric is the **Selective Safety Erasure Index**:

```text
SSEI = safety degradation under cache intervention - ordinary capability degradation under cache intervention
```

For paper claims, SSEI must be paired with:

- prompt-clustered confidence intervals
- paired safety degradation intervals by prompt and seed
- refusal, leakage, and benign over-refusal metrics
- capability controls
- restoration fraction for causal patching
- matched negative controls for non-policy spans

## 4. Planned Results

The paper requires the following tables and figures before submission:

1. Safety vs capability degradation scatter with prompt-clustered intervals.
2. SSEI heatmap by model, suite, policy, and cache budget.
3. Token-role retention heatmap for system, user, template, and generated tokens.
4. Paired safety-degradation forest plot by suite and policy.
5. Causal restoration figure: compressed, patched policy span, patched non-policy span, and policy-pinned mitigation.
6. Failure examples with raw generations and human-audited labels for a small subset.

Current local artifacts are smoke tests only. They must not be interpreted as evidence.

## 5. Discussion

If causal restoration succeeds, the result would imply that safety cannot be evaluated only at training time or prompt-construction time. Serving systems would need alignment-preserving inference policies, including token-role-aware cache retention and cache-compression regression tests for refusal, leakage, and over-refusal. If restoration fails but selective degradation appears, the contribution should be framed as a deployment-evaluation warning rather than a mechanistic erasure claim.

## 6. Limitations

This project intentionally avoids paid or closed-source model judges. Local string and rule metrics are reproducible but coarser than human or strong-judge labels. The paper therefore needs a small blinded human audit subset or a clearly documented open local judge. Current quantization policies are diagnostic simulations, not faithful KVQuant/KIVI implementations. H2O-style attention retention is isolated to diagnostic attention-capture runs to avoid full-sweep memory risk under the 32GB H200 cgroup.

## References

See `paper/references.bib`.
