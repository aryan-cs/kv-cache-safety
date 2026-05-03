import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

import pytest
from check_publication_readiness import (
    _check_active_compression,
    _check_causal_patch_config,
    _check_generation_matrix,
)


def test_generation_matrix_detects_missing_policy_seed_rows() -> None:
    manifest = {
        "cache_policy_labels": ["none", "kv_int4_sim"],
        "seeds": [0, 1],
        "expected_generation_count": 4,
    }
    prompts = [{"suite": "public_refusal_safety", "prompt_id": "p1"}]
    generations = [
        {"suite": "public_refusal_safety", "prompt_id": "p1", "policy": "none", "seed": 0},
        {"suite": "public_refusal_safety", "prompt_id": "p1", "policy": "none", "seed": 1},
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p1",
            "policy": "kv_int4_sim",
            "seed": 0,
        },
    ]
    failures: list[str] = []

    _check_generation_matrix(manifest, prompts, generations, failures)

    assert any("generation row count is 3; expected 4" in failure for failure in failures)
    assert any("missing 1 rows" in failure for failure in failures)


def test_generation_matrix_accepts_complete_grid() -> None:
    manifest = {
        "cache_policy_labels": ["none", "kv_int4_sim"],
        "seeds": [0],
        "expected_generation_count": 2,
    }
    prompts = [{"suite": "public_refusal_safety", "prompt_id": "p1"}]
    generations = [
        {"suite": "public_refusal_safety", "prompt_id": "p1", "policy": "none", "seed": 0},
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p1",
            "policy": "kv_int4_sim",
            "seed": 0,
        },
    ]
    failures: list[str] = []

    _check_generation_matrix(manifest, prompts, generations, failures)

    assert failures == []


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_active_compression_detects_noop_budget(tmp_path: Path) -> None:
    import pandas as pd

    cache_stats = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "sliding_window__budget128",
                "decode_step": 0,
                "original_seq_len": 20,
                "evicted_count": 0,
                "retained_system_tokens": 5,
                "evicted_system_tokens": 0,
                "quantization_bits": None,
                "cache_l2_before": 2.0,
                "cache_l2_after": 2.0,
            }
        ]
    ).to_parquet(cache_stats, index=False)
    failures: list[str] = []

    _check_active_compression(
        cache_stats,
        {"cache_policy_labels": ["none", "sliding_window__budget128"]},
        failures,
    )

    assert any("appears inactive" in failure for failure in failures)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_active_compression_accepts_quantization(tmp_path: Path) -> None:
    import pandas as pd

    cache_stats = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "kv_int4_sim",
                "decode_step": 0,
                "original_seq_len": 20,
                "evicted_count": 0,
                "retained_system_tokens": 5,
                "evicted_system_tokens": 0,
                "quantization_bits": 4,
                "cache_l2_before": 2.0,
                "cache_l2_after": 1.9,
            }
        ]
    ).to_parquet(cache_stats, index=False)
    failures: list[str] = []

    _check_active_compression(cache_stats, {"cache_policy_labels": ["none", "kv_int4_sim"]}, failures)

    assert failures == []


def test_causal_patch_config_requires_system_and_matched_user_control() -> None:
    failures: list[str] = []

    _check_causal_patch_config(
        [
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["system"],
                    "components": ["key", "value"],
                },
            },
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["user"],
                    "match_token_count_to_roles": ["system"],
                    "components": ["key", "value"],
                },
            },
        ],
        failures,
    )

    assert failures == []


def test_causal_patch_config_rejects_fixed_token_only_patch() -> None:
    failures: list[str] = []

    _check_causal_patch_config(
        [{"name": "kv_int4_sim", "patch_from_baseline": {"token_indices": [0, 1]}}],
        failures,
    )

    assert any("role-derived" in failure for failure in failures)
