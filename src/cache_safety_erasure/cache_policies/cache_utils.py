from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for cache policy tensor operations.") from exc
    return torch


def to_legacy_cache(past_key_values: Any) -> tuple[tuple[Any, Any], ...]:
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    return tuple(past_key_values)


def maybe_from_legacy_cache(legacy: tuple[tuple[Any, Any], ...], original: Any) -> Any:
    if hasattr(original, "from_legacy_cache"):
        return original.from_legacy_cache(legacy)
    try:
        from transformers.cache_utils import DynamicCache

        if isinstance(original, DynamicCache):
            return DynamicCache.from_legacy_cache(legacy)
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
) -> tuple[tuple[Any, Any], ...]:
    torch = _torch()
    legacy = to_legacy_cache(past_key_values)
    if not legacy:
        return legacy
    device = legacy[0][0].device
    index = torch.tensor(list(retained_indices), dtype=torch.long, device=device)
    sliced = []
    for key, value in legacy:
        sliced.append((key.index_select(-2, index), value.index_select(-2, index)))
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


def quantize_dequantize_cache(past_key_values: Any, bits: int) -> tuple[tuple[Any, Any], ...]:
    legacy = to_legacy_cache(past_key_values)
    return tuple(
        (quantize_dequantize_tensor(key, bits), quantize_dequantize_tensor(value, bits))
        for key, value in legacy
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
