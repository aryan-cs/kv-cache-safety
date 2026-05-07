from __future__ import annotations

import random
from collections import defaultdict
from math import log
from statistics import mean
from typing import Any

HALDANE_ANSCOMBE_ALPHA = 0.5


def mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if len(clean) == 1:
        value = clean[0]
        return {"mean": value, "ci_low": value, "ci_high": value, "n": 1}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        samples.append(mean(sample))
    alpha = (1.0 - confidence) / 2.0
    samples.sort()
    low_idx = max(0, min(len(samples) - 1, int(alpha * len(samples))))
    high_idx = max(0, min(len(samples) - 1, int((1.0 - alpha) * len(samples)) - 1))
    return {
        "mean": float(mean(clean)),
        "ci_low": float(samples[low_idx]),
        "ci_high": float(samples[high_idx]),
        "n": len(clean),
    }


def paired_delta_ci(
    baseline: dict[tuple[str, int], float],
    treatment: dict[tuple[str, int], float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    keys = sorted(set(baseline).intersection(treatment))
    deltas = [baseline[key] - treatment[key] for key in keys]
    result = mean_ci(deltas, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    result["paired_n"] = len(keys)
    return result


def cluster_mean_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    cluster_key: str = "prompt_id",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for idx, row in enumerate(rows):
        value = row.get(metric)
        if value is None:
            continue
        clusters[str(row.get(cluster_key, f"row_{idx}"))].append(float(value))
    cluster_values = [mean(values) for values in clusters.values() if values]
    result = mean_ci(
        cluster_values,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result["cluster_n"] = len(cluster_values)
    result["row_n"] = sum(len(values) for values in clusters.values())
    return result


def paired_cluster_delta_ci(
    baseline: dict[tuple[str, int], float],
    treatment: dict[tuple[str, int], float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    keys = sorted(set(baseline).intersection(treatment))
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for prompt_id, seed_id in keys:
        _ = seed_id
        by_prompt[prompt_id].append(baseline[(prompt_id, seed_id)] - treatment[(prompt_id, seed_id)])
    cluster_deltas = [mean(values) for values in by_prompt.values() if values]
    result = mean_ci(
        cluster_deltas,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result["paired_n"] = len(keys)
    result["cluster_n"] = len(cluster_deltas)
    return result


def logit_alpha(k: int, n: int, *, alpha: float = HALDANE_ANSCOMBE_ALPHA) -> float | None:
    if n < 0 or k < 0 or k > n:
        raise ValueError("logit_alpha requires 0 <= k <= n")
    if n == 0:
        return None
    return float(log((k + alpha) / (n - k + alpha)))


def binary_worse_event_counts(values: list[float | int | None]) -> dict[str, int]:
    clean = _binary_scores(values)
    return {"k": sum(1 for value in clean if value == 0.0), "n": len(clean)}


def log_odds_degradation_ci(
    baseline_values: list[float | int | None],
    treatment_values: list[float | int | None],
    *,
    alpha: float = HALDANE_ANSCOMBE_ALPHA,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    baseline = _binary_scores(baseline_values)
    treatment = _binary_scores(treatment_values)
    baseline_counts = binary_worse_event_counts(baseline)
    treatment_counts = binary_worse_event_counts(treatment)
    estimate = _log_odds_degradation_from_counts(
        baseline_counts["k"],
        baseline_counts["n"],
        treatment_counts["k"],
        treatment_counts["n"],
        alpha=alpha,
    )
    result = _bootstrap_log_odds_stat(
        [baseline, treatment],
        lambda samples: _log_odds_degradation_from_values(samples[0], samples[1], alpha=alpha),
        estimate,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result.update(
        {
            "baseline_worse_events": baseline_counts["k"],
            "baseline_n": baseline_counts["n"],
            "treatment_worse_events": treatment_counts["k"],
            "treatment_n": treatment_counts["n"],
            "alpha": alpha,
        }
    )
    return result


def ssei_logodds_ci(
    safety_baseline_values: list[float | int | None],
    safety_treatment_values: list[float | int | None],
    capability_baseline_values: list[float | int | None],
    capability_treatment_values: list[float | int | None],
    *,
    alpha: float = HALDANE_ANSCOMBE_ALPHA,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    safety_baseline = _binary_scores(safety_baseline_values)
    safety_treatment = _binary_scores(safety_treatment_values)
    capability_baseline = _binary_scores(capability_baseline_values)
    capability_treatment = _binary_scores(capability_treatment_values)
    estimate = _ssei_logodds_from_values(
        safety_baseline,
        safety_treatment,
        capability_baseline,
        capability_treatment,
        alpha=alpha,
    )
    result = _bootstrap_log_odds_stat(
        [safety_baseline, safety_treatment, capability_baseline, capability_treatment],
        lambda samples: _ssei_logodds_from_values(
            samples[0], samples[1], samples[2], samples[3], alpha=alpha
        ),
        estimate,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result.update(
        {
            "n_safety_baseline": len(safety_baseline),
            "n_safety_treatment": len(safety_treatment),
            "n_capability_baseline": len(capability_baseline),
            "n_capability_treatment": len(capability_treatment),
            "alpha": alpha,
        }
    )
    return result


def _binary_scores(values: list[float | int | None]) -> list[float]:
    clean = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if numeric in {0.0, 1.0}:
            clean.append(numeric)
    return clean


def _log_odds_degradation_from_values(
    baseline: list[float], treatment: list[float], *, alpha: float
) -> float | None:
    baseline_counts = binary_worse_event_counts(baseline)
    treatment_counts = binary_worse_event_counts(treatment)
    return _log_odds_degradation_from_counts(
        baseline_counts["k"],
        baseline_counts["n"],
        treatment_counts["k"],
        treatment_counts["n"],
        alpha=alpha,
    )


def _log_odds_degradation_from_counts(
    baseline_k: int,
    baseline_n: int,
    treatment_k: int,
    treatment_n: int,
    *,
    alpha: float,
) -> float | None:
    baseline_logit = logit_alpha(baseline_k, baseline_n, alpha=alpha)
    treatment_logit = logit_alpha(treatment_k, treatment_n, alpha=alpha)
    if baseline_logit is None or treatment_logit is None:
        return None
    return float(treatment_logit - baseline_logit)


def _ssei_logodds_from_values(
    safety_baseline: list[float],
    safety_treatment: list[float],
    capability_baseline: list[float],
    capability_treatment: list[float],
    *,
    alpha: float,
) -> float | None:
    safety = _log_odds_degradation_from_values(safety_baseline, safety_treatment, alpha=alpha)
    capability = _log_odds_degradation_from_values(
        capability_baseline, capability_treatment, alpha=alpha
    )
    if safety is None or capability is None:
        return None
    return float(safety - capability)


def _bootstrap_log_odds_stat(
    groups: list[list[float]],
    statistic: Any,
    estimate: float | None,
    *,
    confidence: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | None]:
    if estimate is None or any(not group for group in groups):
        return {"mean": estimate, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        resampled = [[group[rng.randrange(len(group))] for _ in group] for group in groups]
        value = statistic(resampled)
        if value is not None:
            samples.append(value)
    if not samples:
        return {"mean": estimate, "ci_low": None, "ci_high": None}
    tail = (1.0 - confidence) / 2.0
    samples.sort()
    low_idx = max(0, min(len(samples) - 1, int(tail * len(samples))))
    high_idx = max(0, min(len(samples) - 1, int((1.0 - tail) * len(samples)) - 1))
    return {
        "mean": float(estimate),
        "ci_low": float(samples[low_idx]),
        "ci_high": float(samples[high_idx]),
    }
