from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    id: str
    suite: str
    user: str
    system: str = ""
    category: str = "general"
    should_refuse: bool | None = None
    expected_answer: str | None = None
    choices: dict[str, str] = field(default_factory=dict)
    hidden_system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptRecord:
        return cls(
            id=str(data["id"]),
            suite=str(data["suite"]),
            user=str(data["user"]),
            system=str(data.get("system", "")),
            category=str(data.get("category", "general")),
            should_refuse=data.get("should_refuse"),
            expected_answer=data.get("expected_answer"),
            choices=dict(data.get("choices", {})),
            hidden_system=data.get("hidden_system"),
            metadata=dict(data.get("metadata", {})),
        )
