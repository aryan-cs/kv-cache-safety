from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cache_safety_erasure.cache_policies.base import CachePolicyDecision
from cache_safety_erasure.cache_policies.cache_utils import (
    cache_layer_count,
    cache_seq_len,
    maybe_from_legacy_cache,
)
from cache_safety_erasure.config import GenerationConfig
from cache_safety_erasure.evals.prompt_record import PromptRecord
from cache_safety_erasure.generation.cache_patching import patch_cache_from_baseline


@dataclass
class GenerationResult:
    text: str
    cache_decisions: list[CachePolicyDecision]


def build_chat_text(tokenizer: Any, prompt: PromptRecord) -> str:
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt.system:
        return f"System: {prompt.system}\nUser: {prompt.user}\nAssistant:"
    return f"User: {prompt.user}\nAssistant:"


def token_roles_for_prompt(tokenizer: Any, prompt: PromptRecord, input_ids: Any) -> list[str]:
    """Best-effort role labels for retained-token logging and policy pinning."""
    total = int(input_ids.shape[-1])
    roles = ["unknown"] * total
    if not prompt.system:
        return ["user"] * total
    system_ids = tokenizer(prompt.system, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    # Chat templates add wrapper tokens, so mark the first approximate system-length region.
    system_len = min(total, max(1, int(system_ids.shape[-1]) + 8))
    for idx in range(system_len):
        roles[idx] = "system"
    for idx in range(system_len, total):
        roles[idx] = "user"
    return roles


def hf_generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    policy: Any,
    generation_config: GenerationConfig,
    patch_from_baseline: dict[str, Any] | None = None,
) -> GenerationResult:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Hugging Face generation.") from exc

    text = build_chat_text(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt")
    device = _model_device(model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    token_roles = token_roles_for_prompt(tokenizer, prompt, input_ids)
    cache_decisions: list[CachePolicyDecision] = []
    generated_ids: list[int] = []

    with torch.inference_mode():
        if int(input_ids.shape[-1]) > 1:
            prefill_ids = input_ids[:, :-1]
            last_prompt_token = input_ids[:, -1:]
            prefill_mask = attention_mask[:, :-1] if attention_mask is not None else None
            outputs = model(
                input_ids=prefill_ids,
                attention_mask=prefill_mask,
                use_cache=True,
                output_attentions=generation_config.capture_attentions,
                return_dict=True,
            )
            baseline_prefill_past = outputs.past_key_values
            past = outputs.past_key_values
            past, decision = policy.apply(
                past,
                step=0,
                token_roles=token_roles[:-1],
                attention_scores=getattr(outputs, "attentions", None),
            )
            if patch_from_baseline:
                past = patch_from_baseline_cache(past, baseline_prefill_past, patch_from_baseline)
                decision.metadata["patched_from_baseline"] = True
            cache_decisions.append(decision)
            outputs = _forward_one_token(
                model=model,
                token_id=last_prompt_token,
                past=past,
                absolute_position=int(input_ids.shape[-1]) - 1,
                output_attentions=generation_config.capture_attentions,
            )
            past = outputs.past_key_values
            past, decision = policy.apply(
                past,
                step=1,
                token_roles=token_roles,
                attention_scores=getattr(outputs, "attentions", None),
            )
            cache_decisions.append(decision)
            absolute_position = int(input_ids.shape[-1])
            decode_step_start = 2
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=generation_config.capture_attentions,
                return_dict=True,
            )
            baseline_prefill_past = outputs.past_key_values
            past = outputs.past_key_values
            past, decision = policy.apply(
                past,
                step=0,
                token_roles=token_roles,
                attention_scores=getattr(outputs, "attentions", None),
            )
            if patch_from_baseline:
                past = patch_from_baseline_cache(past, baseline_prefill_past, patch_from_baseline)
                decision.metadata["patched_from_baseline"] = True
            cache_decisions.append(decision)
            absolute_position = int(input_ids.shape[-1])
            decode_step_start = 1
        next_token = _sample_next_token(outputs.logits[:, -1, :], generation_config)

        for step in range(decode_step_start, decode_step_start + generation_config.max_new_tokens):
            token_id = int(next_token.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)

            cache_len = cache_seq_len(past)
            outputs = _forward_one_token(
                model=model,
                token_id=next_token.reshape(1, 1),
                past=past,
                absolute_position=absolute_position,
                output_attentions=generation_config.capture_attentions,
            )
            absolute_position += 1

            past = outputs.past_key_values
            extended_roles = token_roles + (["generated"] * len(generated_ids))
            past, decision = policy.apply(
                past,
                step=step,
                token_roles=extended_roles,
                attention_scores=getattr(outputs, "attentions", None),
            )
            cache_decisions.append(decision)
            next_token = _sample_next_token(outputs.logits[:, -1, :], generation_config)

            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(stop in partial_text for stop in generation_config.stop_strings):
                break

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    _ = cache_layer_count  # imported for downstream callers and import validation.
    return GenerationResult(text=decoded, cache_decisions=cache_decisions)


def patch_from_baseline_cache(past: Any, baseline_prefill_past: Any, patch_from_baseline: dict[str, Any]) -> Any:
    patched = patch_cache_from_baseline(
        past,
        baseline_prefill_past,
        layers=patch_from_baseline.get("layers"),
        heads=patch_from_baseline.get("heads"),
        token_indices=patch_from_baseline.get("token_indices"),
    )
    return maybe_from_legacy_cache(patched, baseline_prefill_past)


def _forward_one_token(
    *,
    model: Any,
    token_id: Any,
    past: Any,
    absolute_position: int,
    output_attentions: bool,
) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Hugging Face generation.") from exc

    device = token_id.device
    cache_len = cache_seq_len(past)
    kwargs = {
        "input_ids": token_id,
        "attention_mask": torch.ones((1, cache_len + 1), dtype=torch.long, device=device),
        "past_key_values": past,
        "use_cache": True,
        "output_attentions": output_attentions,
        "return_dict": True,
        "position_ids": torch.tensor([[absolute_position]], dtype=torch.long, device=device),
    }
    try:
        kwargs["cache_position"] = torch.tensor([absolute_position], dtype=torch.long, device=device)
        return model(**kwargs)
    except TypeError:
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _sample_next_token(logits: Any, generation_config: GenerationConfig) -> Any:
    import torch

    if generation_config.do_sample and generation_config.temperature > 0:
        scaled = logits / generation_config.temperature
        if generation_config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            mask = cumulative > generation_config.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.argmax(logits, dim=-1)
