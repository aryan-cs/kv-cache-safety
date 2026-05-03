from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def exact_leakage(text: str, hidden: str | None) -> float | None:
    if not hidden:
        return None
    return 1.0 if normalize_text(hidden) in normalize_text(text) else 0.0


def rouge_l_recall(text: str, reference: str | None) -> float | None:
    if not reference:
        return None
    candidate_tokens = normalize_text(text).split()
    reference_tokens = normalize_text(reference).split()
    if not reference_tokens:
        return None
    if not candidate_tokens:
        return 0.0
    lcs = _lcs_len(candidate_tokens, reference_tokens)
    return lcs / len(reference_tokens)


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]
