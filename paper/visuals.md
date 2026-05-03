# Visual Narrative Plan

The paper should not rely only on scatterplots and bar charts. The visual goal is to make cache-mediated safety erasure look like a recognizable phenomenon: a change in transient inference state that produces a selective behavioral shift.

## Figure Families

1. **Cache-state fingerprints**
   - Layer-by-token heatmaps of retained, evicted, or quantized cache state.
   - Token roles shown as colored bands: system/policy, template, user, generated.
   - Intended pattern: policy-relevant spans visibly thin, fracture, or lose cache norm before ordinary user-task spans collapse.

2. **Safety-capability phase portraits**
   - A trajectory for each cache policy as budget tightens.
   - X-axis: ordinary capability degradation.
   - Y-axis: safety/refusal or leakage degradation.
   - Intended pattern: policy trajectories bend upward before moving rightward.

3. **Restoration flow diagrams**
   - Sankey-like or arrow-flow figure from baseline to compressed to patched conditions.
   - Nodes: baseline, compressed, system-patched, matched-user-patched, policy-pinned.
   - Edge color/width: restoration fraction and confidence interval.
   - Intended pattern: system restoration flows back toward baseline more than matched user restoration.

4. **Prompt-level effect constellations**
   - Embed prompt-condition metric vectors or open local text embeddings into 2D.
   - Draw paired edges from baseline to compressed generations for the same prompt.
   - Intended pattern: families of safety prompts move coherently under compression, while capability controls remain tighter.
   - Implemented as `prompt_effect_constellation.*` from paired per-prompt metric deltas.

5. **Safety-state atlas**
   - Small multiples over model, policy, and budget.
   - Each cell combines token-role retention and SSEI intensity.
   - Intended pattern: model/policy regimes where safety erasure is localized versus diffuse.
   - Implemented as `safety_state_atlas.*`, with SSEI as cell color and system/user cache loss as overlaid glyph size.

## Implementation Notes

- Every visual must export PDF, SVG, PNG, source CSV, and manifest hashes.
- Every visual must degrade gracefully if a run lacks causal-patch or attention diagnostics.
- Use smoke output only to validate rendering. Paper figures must come from H200 readiness-passing runs.
- Avoid publishing raw harmful generations in figures. Use redacted high-level labels for audit examples.
