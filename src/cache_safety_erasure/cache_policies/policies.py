from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cache_safety_erasure.cache_policies.base import CachePolicyDecision
from cache_safety_erasure.cache_policies.cache_utils import (
    cache_l2_norm,
    cache_seq_len,
    evicted_from_retained,
    maybe_from_legacy_cache,
    quantize_dequantize_cache,
    slice_legacy_cache,
    to_legacy_cache,
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
        norm = cache_l2_norm(past_key_values) if seq_len else 0.0
        return maybe_from_legacy_cache(to_legacy_cache(past_key_values), past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            tuple(),
            {"cache_l2_before": norm, "cache_l2_after": norm},
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
        before_norm = cache_l2_norm(past_key_values) if seq_len else 0.0
        retained = tuple(range(max(0, seq_len - self.budget), seq_len))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm(sliced) if retained else 0.0
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted_from_retained(seq_len, retained),
            {"cache_l2_before": before_norm, "cache_l2_after": after_norm},
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
        before_norm = cache_l2_norm(past_key_values) if seq_len else 0.0
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            sink_count = min(self.sink_tokens, self.budget, seq_len)
            recent_count = max(0, self.budget - sink_count)
            retained_set = set(range(sink_count))
            retained_set.update(range(max(sink_count, seq_len - recent_count), seq_len))
            retained = tuple(sorted(retained_set))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm(sliced) if retained else 0.0
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted_from_retained(seq_len, retained),
            {
                "sink_tokens": self.sink_tokens,
                "cache_l2_before": before_norm,
                "cache_l2_after": after_norm,
            },
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
        before_norm = cache_l2_norm(past_key_values) if seq_len else 0.0
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            rng = random.Random((self.seed * 1_000_003) + step)
            retained = tuple(sorted(rng.sample(range(seq_len), k=max(0, self.budget))))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm(sliced) if retained else 0.0
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted_from_retained(seq_len, retained),
            {
                "policy_seed": self.seed,
                "cache_l2_before": before_norm,
                "cache_l2_after": after_norm,
            },
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
        before_norm = cache_l2_norm(past_key_values) if seq_len else 0.0
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
        after_norm = cache_l2_norm(sliced) if retained else 0.0
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted_from_retained(seq_len, retained),
            {
                "sink_tokens": self.sink_tokens,
                "recent_tokens": self.recent_tokens,
                "cache_l2_before": before_norm,
                "cache_l2_after": after_norm,
            },
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
        before_norm = cache_l2_norm(past_key_values) if seq_len else 0.0
        quantized = quantize_dequantize_cache(past_key_values, self.bits)
        after_norm = cache_l2_norm(quantized) if seq_len else 0.0
        return maybe_from_legacy_cache(quantized, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            tuple(),
            {
                "quantization_bits": self.bits,
                "cache_l2_before": before_norm,
                "cache_l2_after": after_norm,
            },
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
        before_norm = cache_l2_norm(past_key_values) if seq_len else 0.0
        if self.budget >= seq_len:
            retained = tuple(range(seq_len))
        else:
            protected = set()
            if token_roles:
                protected = {
                    idx
                    for idx, role in enumerate(token_roles[:seq_len])
                    if role in set(self.protected_spans)
                }
            retained_set = set(range(min(self.sink_tokens, seq_len)))
            retained_set.update(protected)
            for idx in range(seq_len - 1, -1, -1):
                if len(retained_set) >= self.budget:
                    break
                retained_set.add(idx)
            if len(retained_set) > self.budget:
                protected_or_sink = {
                    idx for idx in retained_set if idx < self.sink_tokens or idx in protected
                }
                if len(protected_or_sink) <= self.budget:
                    recent_candidates = sorted(
                        (idx for idx in retained_set if idx not in protected_or_sink),
                        reverse=True,
                    )
                    retained_set = set(protected_or_sink)
                    retained_set.update(recent_candidates[: self.budget - len(retained_set)])
            retained = tuple(sorted(retained_set))
        sliced = slice_legacy_cache(past_key_values, retained)
        after_norm = cache_l2_norm(sliced) if retained else 0.0
        return maybe_from_legacy_cache(sliced, past_key_values), CachePolicyDecision(
            self.name,
            step,
            seq_len,
            retained,
            evicted_from_retained(seq_len, retained),
            {
                "protected_spans": ",".join(self.protected_spans),
                "sink_tokens": self.sink_tokens,
                "cache_l2_before": before_norm,
                "cache_l2_after": after_norm,
            },
        )
