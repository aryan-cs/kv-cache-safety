from __future__ import annotations

from cache_safety_erasure.evals.prompt_record import PromptRecord


def character_span_manifest(prompt: PromptRecord) -> list[dict[str, int | str]]:
    """Return simple character-level spans for reproducibility manifests.

    Token-level boundaries depend on the tokenizer and chat template. The HF generator
    separately derives best-effort token roles for cache pinning; this manifest records
    the stable raw prompt components used to produce those roles.
    """
    spans: list[dict[str, int | str]] = []
    cursor = 0
    if prompt.system:
        spans.append({"role": "system", "start": cursor, "end": cursor + len(prompt.system)})
        cursor += len(prompt.system)
    if prompt.user:
        spans.append({"role": "user", "start": cursor, "end": cursor + len(prompt.user)})
    return spans
