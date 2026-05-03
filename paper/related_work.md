# Related Work Notes

## Why We Are Not Centering The Classifier Supply-Chain Idea

The earlier safety-classifier supply-chain audit idea is useful but too much of a synthesis for the user's goal. It combines existing strands rather than revealing a new phenomenon.

Important nearby work:

- Anthropic, "Poisoning Fine-tuning Datasets of Constitutional Classifiers": classifier poisoning is already directly studied.
- Rapid Poison: practical poisoning attacks against rapid-response safety-classifier update pipelines.
- AI-BOM / AIRS / model supply-chain papers: provenance and audit frameworks already exist.
- Guardrail robustness benchmarks: many papers already evaluate open guard models under adversarial prompts.

The repository should cite this line as background and motivation, not as the central contribution.

## Phenomenon-First Inspiration

The target contribution should be closer in spirit to:

- Subliminal Learning: models transmit behavioral traits through semantically unrelated data.
- Token Entanglement in Subliminal Learning: unrelated tokens can causally steer hidden preferences.
- Emergent Misalignment: narrow fine-tuning can induce broad behavioral changes.

Those papers are memorable because they identify surprising mechanisms. Cache-Mediated Safety Erasure aims for the same style: alignment can fail due to transient inference-state optimization.

## Closest Adjacent Work

- The Pitfalls of KV Cache Compression: shows instruction classes degrade unevenly and system prompt leakage can increase under compression.
- Understanding the Physics of Key-Value Cache Compression: frames cache compression as a routing/accessibility perturbation rather than simple token storage.
- MiKV / No Token Left Behind: reports that exhaustive eviction can create safety breaches, hallucinations, and context loss, and that low-precision retention of evicted information can recover degradation.
- CachePrune: edits KV-cache state to reduce indirect prompt injection.
- How Alignment Routes: identifies sparse routing mechanisms for refusal/alignment behavior.

The planned novelty is falsifiable: claim novelty only if patching or pinning specific policy-token cache slices restores safety more than matched non-policy slices. Otherwise, frame the project as a careful open-model replication and extension of KV-compression pitfall work.
