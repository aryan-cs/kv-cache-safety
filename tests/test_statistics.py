from cache_safety_erasure.analysis.statistics import (
    cluster_mean_ci,
    mean_ci,
    paired_cluster_delta_ci,
    paired_delta_ci,
)


def test_mean_ci_is_deterministic() -> None:
    first = mean_ci([0.0, 1.0, 1.0, 0.0], n_bootstrap=100, seed=7)
    second = mean_ci([0.0, 1.0, 1.0, 0.0], n_bootstrap=100, seed=7)
    assert first == second
    assert first["mean"] == 0.5
    assert first["n"] == 4


def test_cluster_mean_ci_resamples_prompt_clusters() -> None:
    rows = [
        {"prompt_id": "a", "score": 1.0},
        {"prompt_id": "a", "score": 0.0},
        {"prompt_id": "b", "score": 1.0},
    ]
    result = cluster_mean_ci(rows, "score", n_bootstrap=100, seed=1)
    assert result["mean"] == 0.75
    assert result["cluster_n"] == 2
    assert result["row_n"] == 3


def test_paired_cluster_delta_ci_averages_seed_repeats_by_prompt() -> None:
    baseline = {("a", 0): 1.0, ("a", 1): 0.8, ("b", 0): 1.0}
    treatment = {("a", 0): 0.0, ("a", 1): 0.2, ("b", 0): 1.0}
    result = paired_cluster_delta_ci(baseline, treatment, n_bootstrap=100, seed=1)
    assert result["mean"] == 0.4
    assert result["paired_n"] == 3
    assert result["cluster_n"] == 2


def test_paired_delta_ci() -> None:
    baseline = {("a", 0): 1.0, ("b", 0): 0.5}
    treatment = {("a", 0): 0.5, ("b", 0): 0.25}
    result = paired_delta_ci(baseline, treatment, n_bootstrap=100, seed=1)
    assert result["mean"] == 0.375
    assert result["paired_n"] == 2
