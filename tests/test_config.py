from pathlib import Path

import pytest

from cache_safety_erasure.config import CachePolicyConfig, parse_experiment_config


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_parse_smoke_config() -> None:
    config, raw = parse_experiment_config(Path("configs/experiments/smoke_mock.yaml"))
    assert raw["run"]["name"] == "smoke_mock"
    assert config.model.provider == "mock"
    assert config.model.allow_cpu_offload is False
    assert config.cache_policies[0].name == "none"
    assert "system_leakage" in config.prompt_suites


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_tiny_hf_smoke_explicitly_allows_offload() -> None:
    config, _raw = parse_experiment_config(Path("configs/experiments/tiny_hf_smoke.yaml"))
    assert config.model.allow_cpu_offload is True


def test_patch_policy_label_includes_components() -> None:
    from cache_safety_erasure.cache_policies.registry import cache_policy_label

    label = cache_policy_label(
        CachePolicyConfig(
            name="kv_int4_sim",
            patch_from_baseline={"components": ["key"], "token_indices": [0, 1, 2]},
        )
    )
    assert label == "kv_int4_sim__patchkey__tok0to2"
