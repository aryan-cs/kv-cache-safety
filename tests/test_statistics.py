from cache_safety_erasure.analysis.statistics import mean_ci, paired_delta_ci


def test_mean_ci_is_deterministic() -> None:
    first = mean_ci([0.0, 1.0, 1.0, 0.0], n_bootstrap=100, seed=7)
    second = mean_ci([0.0, 1.0, 1.0, 0.0], n_bootstrap=100, seed=7)
    assert first == second
    assert first["mean"] == 0.5
    assert first["n"] == 4


def test_paired_delta_ci() -> None:
    baseline = {("a", 0): 1.0, ("b", 0): 0.5}
    treatment = {("a", 0): 0.5, ("b", 0): 0.25}
    result = paired_delta_ci(baseline, treatment, n_bootstrap=100, seed=1)
    assert result["mean"] == 0.375
    assert result["paired_n"] == 2
