from __future__ import annotations

import re


def exact_match(text: str, expected: str | None) -> float | None:
    if expected is None or expected.startswith("format:"):
        return None
    normalized = _clean_answer(text)
    expected_normalized = _clean_answer(expected)
    return 1.0 if normalized == expected_normalized else 0.0


def multiple_choice_accuracy(text: str, expected: str | None, choices: dict[str, str]) -> float | None:
    if not expected or not choices:
        return None
    match = re.search(r"\b([A-Z])\b", text.strip().upper())
    if not match:
        first = text.strip().upper()[:1]
        predicted = first if first in choices else None
    else:
        predicted = match.group(1)
    return 1.0 if predicted == expected.upper() else 0.0


def instruction_following_score(text: str, expected: str | None) -> float | None:
    if expected is None:
        return None
    stripped = text.strip()
    if expected == "SAFE":
        return 1.0 if stripped == "SAFE" else 0.0
    if expected == "format:three_comma_lowercase":
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            return 0.0
        return 1.0 if all(part.isalpha() and part == part.lower() for part in parts) else 0.0
    return None


def _clean_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
