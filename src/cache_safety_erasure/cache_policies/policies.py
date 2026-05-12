from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cache_safety_erasure.cache_policies.base import CachePolicyDecision
from cache_safety_erasure.cache_policies.cache_utils import (
    cache_l2_measurement_mode,
    cache_l2_norm_for_step,
    cache_layer_count,
    cache_seq_len,
    evicted_from_retained,
    maybe_from_legacy_cache,
    quantize_dequantize_cache,
    slice_legacy_cache,
)


@dataclass
class NonePolicy:
    name: str = "none"

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        retained = tuple(range(seq_len))
        norm = cache_l2_norm_for_step(past_key_values, step)
        evicted = tuple()
        return past_key_values, CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(past_key_values, norm, norm, retained, evicted, token_roles),
        )


@dataclass
class SlidingWindowPolicy:
    budget: int
    name: str = "sliding_window"

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        before_norm = cache_l2_norm_for_step(past_key_values, step)
        retained = tuple(range(max(0, seq_len - self.budget), seq_len))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm_for_step(sliced, step) if retained else _empty_l2_value(before_norm)
        evicted = evicted_from_retained(seq_len, retained)
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(past_key_values, before_norm, after_norm, retained, evicted, token_roles),
        )


@dataclass
class SinkRecentPolicy:
    budget: int
    sink_tokens: int = 4
    name: str = "sink_recent"

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        before_norm = cache_l2_norm_for_step(past_key_values, step)
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            sink_count = min(self.sink_tokens, self.budget, seq_len)
            recent_count = max(0, self.budget - sink_count)
            retained_set = set(range(sink_count))
            retained_set.update(range(max(sink_count, seq_len - recent_count), seq_len))
            retained = tuple(sorted(retained_set))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm_for_step(sliced, step) if retained else _empty_l2_value(before_norm)
        evicted = evicted_from_retained(seq_len, retained)
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(
                past_key_values,
                before_norm,
                after_norm,
                retained,
                evicted,
                token_roles,
                sink_tokens=self.sink_tokens,
            ),
        )


@dataclass
class RandomMatchedPolicy:
    budget: int
    seed: int = 0
    name: str = "random_matched"

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        before_norm = cache_l2_norm_for_step(past_key_values, step)
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            rng = random.Random((self.seed * 1_000_003) + step)
            retained = tuple(sorted(rng.sample(range(seq_len), k=max(0, self.budget))))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm_for_step(sliced, step) if retained else _empty_l2_value(before_norm)
        evicted = evicted_from_retained(seq_len, retained)
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(
                past_key_values,
                before_norm,
                after_norm,
                retained,
                evicted,
                token_roles,
                policy_seed=self.seed,
            ),
        )


@dataclass
class AttentionH2OPolicy:
    budget: int
    sink_tokens: int = 4
    recent_tokens: int | None = None
    name: str = "attention_h2o"

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        before_norm = cache_l2_norm_for_step(past_key_values, step)
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            recent_count = self.recent_tokens if self.recent_tokens is not None else max(1, self.budget // 4)
            sink_count = min(self.sink_tokens, self.budget)
            retained_set = set(range(min(sink_count, seq_len)))
            retained_set.update(range(max(0, seq_len - recent_count), seq_len))
            remaining_budget = max(0, self.budget - len(retained_set))
            candidates = [i for i in range(seq_len) if i not in retained_set]
            if attention_scores is not None and candidates and remaining_budget:
                scores = _attention_scores_to_list(attention_scores, seq_len)
                candidates = sorted(candidates, key=lambda idx: scores[idx], reverse=True)
            retained_set.update(candidates[:remaining_budget])
            retained = tuple(sorted(retained_set))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm_for_step(sliced, step) if retained else _empty_l2_value(before_norm)
        evicted = evicted_from_retained(seq_len, retained)
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(
                past_key_values,
                before_norm,
                after_norm,
                retained,
                evicted,
                token_roles,
                sink_tokens=self.sink_tokens,
                recent_tokens=self.recent_tokens,
                attention_scores_used=attention_scores is not None,
            ),
        )


def _attention_scores_to_list(attention_scores: Any, seq_len: int) -> list[float]:
    try:
        import torch
    except ModuleNotFoundError:
        return [0.0] * seq_len
    if isinstance(attention_scores, (list, tuple)):
        tensor = attention_scores[-1]
    else:
        tensor = attention_scores
    if tensor is None:
        return [0.0] * seq_len
    with torch.no_grad():
        # Expected shape: batch, heads, query_len, key_len.
        reduced = tensor.detach().float().mean(dim=0).mean(dim=0).mean(dim=0)
        values = reduced[-seq_len:].cpu().tolist()
    if len(values) < seq_len:
        values = ([0.0] * (seq_len - len(values))) + values
    return [float(x) for x in values[:seq_len]]


@dataclass
class QuantizedCachePolicy:
    bits: int
    name: str

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        retained = tuple(range(seq_len))
        before_norm = cache_l2_norm_for_step(past_key_values, step)
        quantized = quantize_dequantize_cache(past_key_values, self.bits)
        after_norm = cache_l2_norm_for_step(quantized, step)
        evicted = tuple()
        return maybe_from_legacy_cache(quantized, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(
                past_key_values,
                before_norm,
                after_norm,
                retained,
                evicted,
                token_roles,
                quantization_bits=self.bits,
            ),
        )


@dataclass
class PolicyPinnedPolicy:
    budget: int
    protected_spans: tuple[str, ...] = ("system", "policy")
    sink_tokens: int = 4
    name: str = "policy_pinned"

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        seq_len = cache_seq_len(past_key_values)
        before_norm = cache_l2_norm_for_step(past_key_values, step)
        protected: set[int] = set()
        if token_roles:
            protected = {
                idx
                for idx, role in enumerate(token_roles[:seq_len])
                if role in set(self.protected_spans)
            }
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            retained_set = set(sorted(protected)[: self.budget])
            if len(retained_set) < self.budget:
                retained_set.update(range(min(self.sink_tokens, seq_len)))
            for idx in range(seq_len - 1, -1, -1):
                if len(retained_set) >= self.budget:
                    break
                retained_set.add(idx)
            if len(retained_set) > self.budget:
                priority = sorted(
                    retained_set,
                    key=lambda idx: (idx not in protected, idx if idx in protected else -idx),
                )
                retained_set = set(priority[: self.budget])
            retained = tuple(sorted(retained_set))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm_for_step(sliced, step) if retained else _empty_l2_value(before_norm)
        evicted = evicted_from_retained(seq_len, retained)
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted,
            _decision_metadata(
                past_key_values,
                before_norm,
                after_norm,
                retained,
                evicted,
                token_roles,
                protected_spans=",".join(self.protected_spans),
                sink_tokens=self.sink_tokens,
                protected_candidate_count=len(protected),
                protected_retained_count=len(set(retained).intersection(protected)),
                protected_dropped_count=len(protected.difference(retained)),
            ),
        )


def _decision_metadata(
    past_key_values: Any,
    before_norm: float | None,
    after_norm: float | None,
    retained: tuple[int, ...],
    evicted: tuple[int, ...],
    token_roles: list[str] | None,
    **extra: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "layer_count": cache_layer_count(past_key_values),
        "cache_l2_before": before_norm,
        "cache_l2_after": after_norm,
        "cache_l2_measurement": cache_l2_measurement_mode(),
    }
    metadata.update(_role_count_metadata("retained", retained, token_roles))
    metadata.update(_role_count_metadata("evicted", evicted, token_roles))
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def _role_count_metadata(
    prefix: str, indices: tuple[int, ...], token_roles: list[str] | None
) -> dict[str, int]:
    if not token_roles:
        return {}
    counts: dict[str, int] = {}
    for idx in indices:
        role = token_roles[idx] if 0 <= idx < len(token_roles) else "unknown"
        safe_role = role.replace("-", "_")
        counts[f"{prefix}_{safe_role}_tokens"] = counts.get(f"{prefix}_{safe_role}_tokens", 0) + 1
    return counts


def _empty_l2_value(reference_norm: float | None) -> float | None:
    return 0.0 if reference_norm is not None else None
