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
    hf_device_map = getattr(model, "hf_device_map", None)
    if not config.allow_cpu_offload and _device_map_has_cpu_or_disk(hf_device_map):
        raise RuntimeError(
            "Model loaded with CPU or disk offload in `hf_device_map`, which is disabled for "
            "paper evidence runs. Use a smaller model, lower precision, or set "
            "`model.allow_cpu_offload: true` only for local smoke tests."
        )
    model.eval()
    return ModelBundle(provider="hf", model_id=config.model_id, model=model, tokenizer=tokenizer)


def hf_device_map(model: Any) -> dict[str, str] | None:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return None
    return {str(module): str(device) for module, device in device_map.items()}


def _device_map_has_cpu_or_disk(device_map: Any) -> bool:
    if not isinstance(device_map, dict):
        return False
    for device in device_map.values():
        device_name = str(device).lower()
        if device_name == "cpu" or "disk" in device_name:
            return True
    return False
