from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cache_safety_erasure.config import ModelConfig
from cache_safety_erasure.models.mock import MockModelBundle


@dataclass
class ModelBundle:
    provider: str
    model_id: str
    model: Any
    tokenizer: Any


def load_model(config: ModelConfig) -> ModelBundle | MockModelBundle:
    if config.provider == "mock" or config.model_id.startswith("mock://"):
        return MockModelBundle(model_id=config.model_id)
    if config.provider != "hf":
        raise ValueError(f"Unsupported model provider: {config.provider}")
    return _load_hf_model(config)


def _load_hf_model(config: ModelConfig) -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch and transformers are required for Hugging Face model runs. "
            "Run `uv sync --extra dev` in this repository."
        ) from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported dtype `{config.dtype}`. Use float32, float16, or bfloat16.")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
        "local_files_only": config.local_files_only,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    model.eval()
    return ModelBundle(provider="hf", model_id=config.model_id, model=model, tokenizer=tokenizer)
