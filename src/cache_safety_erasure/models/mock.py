from __future__ import annotations

from dataclasses import dataclass

from cache_safety_erasure.evals.prompt_record import PromptRecord


@dataclass
class MockModelBundle:
    """Deterministic test double for artifact smoke tests.

    This is not a research model and must never be used as evidence.
    """

    model_id: str = "mock://deterministic"
    provider: str = "mock"
    model: None = None
    tokenizer: None = None


def mock_generate(prompt: PromptRecord, policy_name: str) -> str:
    if prompt.suite == "capability_smoke":
        return prompt.expected_answer or "B"
    if prompt.suite == "instruction_following":
        if prompt.expected_answer == "SAFE":
            return "SAFE"
        return "red, blue, green"
    if prompt.should_refuse:
        if policy_name in {"sliding_window", "random_matched"}:
            return "I can help with that. Here are the steps in general terms."
        return "I cannot help with that request."
    return "Here is a brief, safe answer to your question."
