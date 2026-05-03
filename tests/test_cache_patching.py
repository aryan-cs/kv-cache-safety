import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_patch_cache_from_baseline_can_patch_key_only() -> None:
    import torch

    from cache_safety_erasure.generation.cache_patching import patch_cache_from_baseline

    target = ((torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),)
    baseline = ((torch.ones(1, 2, 3, 4), torch.full((1, 2, 3, 4), 2.0)),)
    patched = patch_cache_from_baseline(target, baseline, heads=[1], token_indices=[2], components=["key"])
    assert patched[0][0][0, 1, 2, 0].item() == 1.0
    assert patched[0][1][0, 1, 2, 0].item() == 0.0
    assert patched[0][0][0, 0, 2, 0].item() == 0.0


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_patch_cache_from_baseline_rejects_unknown_component() -> None:
    import torch

    from cache_safety_erasure.generation.cache_patching import patch_cache_from_baseline

    cache = ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),)
    with pytest.raises(ValueError, match="Unknown cache patch components"):
        patch_cache_from_baseline(cache, cache, components=["query"])


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_resolve_patch_spec_selects_role_matched_tokens() -> None:
    import torch

    from cache_safety_erasure.generation.cache_patching import resolve_patch_from_baseline_spec

    cache = ((torch.zeros(1, 1, 6, 1), torch.zeros(1, 1, 6, 1)),)
    resolved, metadata = resolve_patch_from_baseline_spec(
        {
            "token_roles": ["user"],
            "match_token_count_to_roles": ["system"],
            "max_tokens": 2,
            "selection": "first",
        },
        token_roles=["system", "system", "template", "user", "user", "user"],
        target_cache=cache,
        baseline_cache=cache,
    )

    assert resolved["token_indices"] == [3, 4]
    assert metadata["patched_token_count"] == 2
    assert metadata["patched_roles"] == "user"
    assert metadata["patch_matched_roles"] == "system"
