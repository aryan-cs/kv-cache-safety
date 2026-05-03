import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from check_publication_readiness import _check_generation_matrix


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
