from __future__ import annotations

import os
from typing import Any


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for cache policy tensor operations.") from exc
    return torch


def to_legacy_cache(past_key_values: Any) -> tuple[tuple[Any, ...], ...]:
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    return tuple(past_key_values)


def maybe_from_legacy_cache(legacy: tuple[tuple[Any, ...], ...], original: Any) -> Any:
    if isinstance(original, tuple):
        return legacy
    if isinstance(original, list):
        return list(legacy)
    if hasattr(original, "from_legacy_cache"):
        return original.from_legacy_cache(legacy)
    try:
        from transformers.cache_utils import DynamicCache

        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy)
        return DynamicCache(legacy)
    except Exception:
        pass
    return legacy


def cache_seq_len(past_key_values: Any) -> int:
    legacy = to_legacy_cache(past_key_values)
    if not legacy:
        return 0
    key = legacy[0][0]
    return int(key.shape[-2])


def cache_layer_count(past_key_values: Any) -> int:
    return len(to_legacy_cache(past_key_values))


def slice_legacy_cache(
    past_key_values: Any, retained_indices: list[int] | tuple[int, ...]
) -> tuple[tuple[Any, ...], ...]:
    torch = _torch()
    legacy = to_legacy_cache(past_key_values)
    if not legacy:
        return legacy
    device = legacy[0][0].device
    index = torch.tensor(list(retained_indices), dtype=torch.long, device=device)
    sliced = []
    for layer in legacy:
        key, value, *rest = layer
        sliced.append((key.index_select(-2, index), value.index_select(-2, index), *rest))
    return tuple(sliced)


def quantize_dequantize_tensor(tensor: Any, bits: int) -> Any:
    if bits not in {4, 8}:
        raise ValueError(f"Only 4-bit and 8-bit simulation are supported, got {bits}.")
    qmax = (2 ** (bits - 1)) - 1
    max_abs = tensor.detach().abs().amax()
    if float(max_abs) == 0.0:
        return tensor.clone()
    scale = max_abs / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax)
    return (quantized * scale).to(dtype=tensor.dtype)


def quantize_dequantize_cache(past_key_values: Any, bits: int) -> tuple[tuple[Any, ...], ...]:
    legacy = to_legacy_cache(past_key_values)
    return tuple(
        (
            quantize_dequantize_tensor(layer[0], bits),
            quantize_dequantize_tensor(layer[1], bits),
            *layer[2:],
        )
        for layer in legacy
    )


def cache_l2_norm(past_key_values: Any) -> float:
    torch = _torch()
    total = torch.tensor(0.0)
    for layer in to_legacy_cache(past_key_values):
        key, value = layer[0], layer[1]
        total = total + key.detach().float().pow(2).sum().cpu()
        total = total + value.detach().float().pow(2).sum().cpu()
    return float(total.sqrt().item())


def cache_l2_norm_for_step(past_key_values: Any, step: int, *, empty_value: float = 0.0) -> float | None:
    if not should_measure_cache_l2(step):
        return None
    if cache_seq_len(past_key_values) == 0:
        return empty_value
    return cache_l2_norm(past_key_values)


def cache_l2_measurement_mode() -> str:
    return os.environ.get("CACHE_SAFETY_CACHE_L2_STEPS", "prefill").strip().lower()


def should_measure_cache_l2(step: int) -> bool:
    mode = cache_l2_measurement_mode()
    if mode in {"all", "1", "true", "yes"}:
        return True
    if mode in {"none", "0", "false", "no", "off"}:
        return False
    if mode in {"prefill", "pre_response", "pre-response", ""}:
        return step <= 1
    raise ValueError(
        "CACHE_SAFETY_CACHE_L2_STEPS must be one of all, prefill, or none; "
        f"got {mode!r}"
    )


def indices_for_budget(seq_len: int, budget: int | None) -> list[int]:
    if budget is None or budget >= seq_len:
        return list(range(seq_len))
    if budget <= 0:
        return []
    return list(range(seq_len - budget, seq_len))


def evicted_from_retained(seq_len: int, retained: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    retained_set = set(retained)
    return tuple(i for i in range(seq_len) if i not in retained_set)
