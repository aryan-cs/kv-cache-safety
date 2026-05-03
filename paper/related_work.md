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
- CachePrune: edits KV-cache state to reduce indirect prompt injection.
- How Alignment Routes: identifies sparse routing mechanisms for refusal/alignment behavior.

The planned novelty is the causal bridge: KV-cache optimization may selectively erase alignment-routing state while preserving general capability.
