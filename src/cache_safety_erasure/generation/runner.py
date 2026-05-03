from __future__ import annotations

from cache_safety_erasure.config import GenerationConfig
from cache_safety_erasure.evals.prompt_record import PromptRecord
from cache_safety_erasure.generation.hf_generate import GenerationResult, hf_generate
from cache_safety_erasure.models.mock import MockModelBundle, mock_generate


def generate_one(
    *,
    model_bundle: object,
    prompt: PromptRecord,
    policy: object,
    generation_config: GenerationConfig,
    patch_from_baseline: dict | None = None,
) -> GenerationResult:
    if isinstance(model_bundle, MockModelBundle):
        text = mock_generate(prompt, getattr(policy, "name", "none"))
        return GenerationResult(text=text, cache_decisions=[])
    return hf_generate(
        model=model_bundle.model,
        tokenizer=model_bundle.tokenizer,
        prompt=prompt,
        policy=policy,
        generation_config=generation_config,
        patch_from_baseline=patch_from_baseline,
    )
