from __future__ import annotations

from cache_safety_erasure.cache_policies.policies import (
    AttentionH2OPolicy,
    NonePolicy,
    PolicyPinnedPolicy,
    QuantizedCachePolicy,
    RandomMatchedPolicy,
    SinkRecentPolicy,
    SlidingWindowPolicy,
)
from cache_safety_erasure.config import CachePolicyConfig


def _require_budget(config: CachePolicyConfig) -> int:
    if config.budget is None:
        raise ValueError(f"Cache policy `{config.name}` requires `budget`.")
    return config.budget


def build_cache_policy(config: CachePolicyConfig, seed: int) -> object:
    name = config.name
    label = cache_policy_label(config)
    if name == "none":
        return NonePolicy()
    if name == "sliding_window":
        return SlidingWindowPolicy(budget=_require_budget(config), name=label)
    if name == "sink_recent":
        return SinkRecentPolicy(budget=_require_budget(config), sink_tokens=config.sink_tokens, name=label)
    if name == "random_matched":
        return RandomMatchedPolicy(
            budget=_require_budget(config),
            seed=config.seed if config.seed is not None else seed,
            name=label,
        )
    if name == "attention_h2o":
        return AttentionH2OPolicy(
            budget=_require_budget(config),
            sink_tokens=config.sink_tokens,
            recent_tokens=config.recent_tokens,
            name=label,
        )
    if name == "kv_int8_sim":
        return QuantizedCachePolicy(bits=8, name=name)
    if name == "kv_int4_sim":
        return QuantizedCachePolicy(bits=4, name=name)
    if name == "policy_pinned":
        return PolicyPinnedPolicy(
            budget=_require_budget(config),
            protected_spans=config.protected_spans,
            sink_tokens=config.sink_tokens,
            name=label,
        )
    raise ValueError(f"Unknown cache policy: {name}")


def cache_policy_label(config: CachePolicyConfig) -> str:
    if config.name == "none":
        return "none"
    parts = [config.name]
    if config.budget is not None:
        parts.append(f"budget{config.budget}")
    if config.name in {"sink_recent", "attention_h2o", "policy_pinned"}:
        parts.append(f"sink{config.sink_tokens}")
    if config.recent_tokens is not None:
        parts.append(f"recent{config.recent_tokens}")
    if config.name == "random_matched" and config.seed is not None:
        parts.append(f"seed{config.seed}")
    return "__".join(parts)
