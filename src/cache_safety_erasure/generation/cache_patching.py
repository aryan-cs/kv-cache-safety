from __future__ import annotations

from typing import Any

from cache_safety_erasure.cache_policies.cache_utils import cache_seq_len, to_legacy_cache


def patch_cache_from_baseline(
    target_cache: Any,
    baseline_cache: Any,
    *,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    token_indices: list[int] | None = None,
    components: list[str] | None = None,
) -> tuple[tuple[Any, Any], ...]:
    """Patch selected K/V slices from an uncompressed baseline cache into target cache.

    This operates on retained cache positions. If a token was evicted from the compressed
    target cache, there is no destination slot; mitigation policies such as `policy_pinned`
    are the intended way to preserve such tokens.
    """
    target = to_legacy_cache(target_cache)
    baseline = to_legacy_cache(baseline_cache)
    layer_set = set(layers if layers is not None else range(len(target)))
    component_set = set(components or ["key", "value"])
    unknown_components = component_set.difference({"key", "value"})
    if unknown_components:
        raise ValueError(f"Unknown cache patch components: {sorted(unknown_components)}")
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
                if "key" in component_set:
                    new_k[:, head, token_idx, :] = base_k[:, head, token_idx, :]
                if "value" in component_set:
                    new_v[:, head, token_idx, :] = base_v[:, head, token_idx, :]
        patched.append((new_k, new_v, *target_rest))
    return tuple(patched)


def resolve_patch_from_baseline_spec(
    patch_spec: dict[str, Any],
    *,
    token_roles: list[str] | None,
    target_cache: Any,
    baseline_cache: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve role-based patch specs to concrete token indices.

    Role-derived patching is intended for length-preserving interventions such as
    cache quantization. Eviction policies can delete destination slots, so restoration
    claims should rely on `policy_pinned` mitigation or separate fixed-context probes.
    """
    resolved = dict(patch_spec)
    target_seq = cache_seq_len(target_cache)
    baseline_seq = cache_seq_len(baseline_cache)
    limit = min(target_seq, baseline_seq, len(token_roles or []))
    requested_roles = _string_list(
        patch_spec.get("token_roles") or patch_spec.get("roles") or patch_spec.get("role")
    )
    matched_roles = _string_list(
        patch_spec.get("match_token_count_to_roles")
        or patch_spec.get("matched_token_roles")
        or patch_spec.get("match_roles")
    )
    if "token_indices" not in resolved and requested_roles:
        candidates = [
            idx
            for idx, role in enumerate((token_roles or [])[:limit])
            if role in set(requested_roles)
        ]
        if matched_roles:
            match_count = sum(
                1 for role in (token_roles or [])[:limit] if role in set(matched_roles)
            )
            max_tokens = patch_spec.get("max_tokens")
            if max_tokens is not None:
                match_count = min(match_count, int(max_tokens))
            indices = _select_indices(candidates, match_count, str(patch_spec.get("selection", "first")))
        else:
            max_tokens = patch_spec.get("max_tokens")
            indices = (
                _select_indices(candidates, int(max_tokens), str(patch_spec.get("selection", "first")))
                if max_tokens is not None
                else candidates
            )
        resolved["token_indices"] = indices

    token_indices = resolved.get("token_indices")
    token_count = len(token_indices) if token_indices is not None else min(target_seq, baseline_seq)
    metadata = {
        "patched_from_baseline": True,
        "patched_token_count": token_count,
        "patched_roles": ",".join(requested_roles),
        "patch_matched_roles": ",".join(matched_roles),
        "patched_token_indices": ",".join(str(idx) for idx in token_indices or []),
        "patch_selection": str(patch_spec.get("selection", "")),
        "patch_layers": ",".join(str(layer) for layer in _string_list(patch_spec.get("layers"))),
        "patch_heads": ",".join(str(head) for head in _string_list(patch_spec.get("heads"))),
        "patch_components": ",".join(
            str(component) for component in _string_list(patch_spec.get("components") or ["key", "value"])
        ),
    }
    return resolved, metadata


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _select_indices(indices: list[int], count: int, selection: str) -> list[int]:
    if count <= 0:
        return []
    if len(indices) <= count:
        return list(indices)
    if selection == "last":
        return indices[-count:]
    if selection == "middle":
        start = max(0, (len(indices) - count) // 2)
        return indices[start : start + count]
    return indices[:count]
