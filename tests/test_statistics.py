from cache_safety_erasure.analysis.statistics import (
    binary_worse_event_counts,
    cluster_mean_ci,
    log_odds_degradation_ci,
    logit_alpha,
    mean_ci,
    paired_cluster_delta_ci,
    paired_delta_ci,
    ssei_logodds_ci,
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


def test_logit_alpha_uses_registered_haldane_anscombe_smoothing() -> None:
    assert round(logit_alpha(0, 4), 6) == round(-2.1972245773362196, 6)
    assert binary_worse_event_counts([1.0, 0.0, None, 0.5, 0.0]) == {"k": 2, "n": 3}


def test_log_odds_degradation_ci_counts_binary_worse_events() -> None:
    result = log_odds_degradation_ci(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        n_bootstrap=100,
        seed=3,
    )

    assert result["baseline_worse_events"] == 0
    assert result["treatment_worse_events"] == 3
    assert result["alpha"] == 0.5
    assert result["mean"] is not None
    assert result["ci_low"] is not None


def test_ssei_logodds_ci_subtracts_capability_logodds_degradation() -> None:
    result = ssei_logodds_ci(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        n_bootstrap=100,
        seed=3,
    )

    assert result["mean"] is not None
    assert result["mean"] > 0
    assert result["n_safety_baseline"] == 4
    assert result["n_capability_treatment"] == 4
