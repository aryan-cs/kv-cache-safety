from __future__ import annotations

from cache_safety_erasure.evals.prompt_record import PromptRecord
from cache_safety_erasure.evals.rendering import raw_component_char_spans


def character_span_manifest(prompt: PromptRecord) -> list[dict[str, int | str]]:
    """Return simple character-level spans for reproducibility manifests.

    Token-level boundaries depend on the tokenizer and chat template. The HF generator
    separately derives best-effort token roles for cache pinning; this manifest records
    the stable raw prompt components used to produce those roles.
    """
    return raw_component_char_spans(prompt)
