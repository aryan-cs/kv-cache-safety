from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any


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
