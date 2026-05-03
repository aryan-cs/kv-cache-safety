from __future__ import annotations

from typing import Any

from cache_safety_erasure.cache_policies.cache_utils import to_legacy_cache


def patch_cache_from_baseline(
    target_cache: Any,
    baseline_cache: Any,
    *,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    token_indices: list[int] | None = None,
) -> tuple[tuple[Any, Any], ...]:
    """Patch selected K/V slices from an uncompressed baseline cache into target cache.

    This operates on retained cache positions. If a token was evicted from the compressed
    target cache, there is no destination slot; mitigation policies such as `policy_pinned`
    are the intended way to preserve such tokens.
    """
    target = to_legacy_cache(target_cache)
    baseline = to_legacy_cache(baseline_cache)
    layer_set = set(layers if layers is not None else range(len(target)))
    patched = []
    for layer_idx, (target_layer, base_layer) in enumerate(zip(target, baseline, strict=False)):
        target_k, target_v, *target_rest = target_layer
        base_k, base_v = base_layer[0], base_layer[1]
        if layer_idx not in layer_set:
            patched.append(target_layer)
            continue
        new_k = target_k.clone()
        new_v = target_v.clone()
        target_seq = new_k.shape[-2]
        base_seq = base_k.shape[-2]
        token_set = token_indices if token_indices is not None else list(range(min(target_seq, base_seq)))
        head_set = heads if heads is not None else list(range(new_k.shape[1]))
        for head in head_set:
            if head >= new_k.shape[1] or head >= base_k.shape[1]:
                continue
            for token_idx in token_set:
                if token_idx >= target_seq or token_idx >= base_seq:
                    continue
                new_k[:, head, token_idx, :] = base_k[:, head, token_idx, :]
                new_v[:, head, token_idx, :] = base_v[:, head, token_idx, :]
        patched.append((new_k, new_v, *target_rest))
    return tuple(patched)
