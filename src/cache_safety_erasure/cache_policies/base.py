from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CachePolicyDecision:
    policy_name: str
    step: int
    original_seq_len: int
    retained_indices: tuple[int, ...]
    evicted_indices: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retained_count(self) -> int:
        return len(self.retained_indices)

    @property
    def evicted_count(self) -> int:
        return len(self.evicted_indices)

    def to_rows(self, prompt_id: str, seed: int, layer_count: int | None = None) -> list[dict[str, Any]]:
        if layer_count is None and self.metadata.get("layer_count") is not None:
            try:
                layer_count = int(self.metadata["layer_count"])
            except (TypeError, ValueError):
                layer_count = None
        base = {
            "prompt_id": prompt_id,
            "seed": seed,
            "policy": self.policy_name,
            "decode_step": self.step,
            "original_seq_len": self.original_seq_len,
            "retained_count": self.retained_count,
            "evicted_count": self.evicted_count,
            "retained_indices": ",".join(str(x) for x in self.retained_indices),
            "evicted_indices": ",".join(str(x) for x in self.evicted_indices),
            **self.metadata,
        }
        if layer_count is None:
            return [base]
        return [{**base, "layer": layer} for layer in range(layer_count)]


class CachePolicy(Protocol):
    name: str

    def apply(
        self,
        past_key_values: Any,
        *,
        step: int,
        token_roles: list[str] | None = None,
        attention_scores: Any | None = None,
    ) -> tuple[Any, CachePolicyDecision]:
        """Return possibly modified cache and a logging decision."""
