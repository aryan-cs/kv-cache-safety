from pathlib import Path

import pytest

from cache_safety_erasure.config import parse_experiment_config


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_parse_smoke_config() -> None:
    config, raw = parse_experiment_config(Path("configs/experiments/smoke_mock.yaml"))
    assert raw["run"]["name"] == "smoke_mock"
    assert config.model.provider == "mock"
    assert config.cache_policies[0].name == "none"
    assert "system_leakage" in config.prompt_suites
