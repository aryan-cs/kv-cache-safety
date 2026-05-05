import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from merge_ci_extension_results import merge_ci_extension_results

from cache_safety_erasure.utils.io import read_jsonl, write_json, write_jsonl, write_parquet


def test_merge_ci_extension_adds_only_new_prompt_clusters(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    extension = tmp_path / "extension"
    output = tmp_path / "merged"
    _write_run(
        primary,
        prompts=[_prompt("public_refusal_safety", "p1")],
        policies=["none", "kv_int4_sim"],
    )
    _write_run(
        extension,
        prompts=[_prompt("public_refusal_safety", "p1"), _prompt("public_refusal_safety", "p2")],
        policies=["none", "kv_int4_sim"],
    )

    summary = merge_ci_extension_results(
        primary_results_dir=primary,
        extension_results_dir=extension,
        output_results_dir=output,
    )

    assert summary["added_prompt_count"] == 1
    assert summary["skipped_duplicate_prompt_count"] == 1
    generations = read_jsonl(output / "generations.jsonl")
    assert {(row["prompt_id"], row["policy"]) for row in generations} == {
        ("p1", "none"),
        ("p1", "kv_int4_sim"),
        ("p2", "none"),
        ("p2", "kv_int4_sim"),
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_name"] == "merged"
    assert manifest["prompt_counts"] == {"public_refusal_safety": 2}
    assert manifest["expected_generation_count"] == 4
    assert manifest["combined_results"]["base_run_name"] == "h200_qwen_full_sweep"
    assert manifest["combined_results"]["merged_run_name"] == "merged"
    assert manifest["combined_results"]["output_results_dir"] == str(output)
    assert manifest["combined_results"]["skipped_duplicate_prompt_count"] == 1
    assert (output / "metrics.json").exists()
    assert (output / "cache_stats.parquet").exists()


def test_merge_ci_extension_requires_same_policy_matrix(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    extension = tmp_path / "extension"
    _write_run(
        primary,
        prompts=[_prompt("public_refusal_safety", "p1")],
        policies=["none", "kv_int4_sim"],
    )
    _write_run(
        extension,
        prompts=[_prompt("public_refusal_safety", "p2")],
        policies=["none"],
    )

    with pytest.raises(SystemExit, match="Cannot merge incompatible runs"):
        merge_ci_extension_results(
            primary_results_dir=primary,
            extension_results_dir=extension,
            output_results_dir=tmp_path / "merged",
        )


def _write_run(path: Path, *, prompts: list[dict], policies: list[str]) -> None:
    path.mkdir()
    (path / "figures").mkdir()
    write_json(
        path / "manifest.json",
        {
            "run_name": "h200_qwen_full_sweep",
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "model_provider": "hf",
            "git_commit": "abc",
            "git_dirty": False,
            "config_sha256": "def",
            "prompt_suites": sorted({prompt["suite"] for prompt in prompts}),
            "prompt_counts": _prompt_counts(prompts),
            "prompt_suite_manifests": {},
            "cache_policy_configs": [{"name": policy} for policy in policies],
            "cache_policy_labels": policies,
            "seeds": [0],
            "expected_generation_count": len(prompts) * len(policies),
        },
    )
    write_json(path / "environment.json", {"git_commit": "abc", "torch_cuda_available": True})
    (path / "config.resolved.yaml").write_text("run:\n  name: h200_qwen_full_sweep\n", encoding="utf-8")
    prompt_rows = [
        {
            **prompt,
            "category": "harmful",
            "system": "Refuse unsafe requests.",
            "should_refuse": True,
            "metadata": {
                "source_dataset": "dataset",
                "source_split": "test",
                "source_revision": "rev",
                "source_fingerprint": "fingerprint",
                "source_version": "1.0.0",
                "source_config": None,
                "source_config_name": "default",
                "source_homepage": "",
                "source_license": "",
                "source_row_index": int(prompt["prompt_id"][1:]),
            },
        }
        for prompt in prompts
    ]
    write_jsonl(path / "prompts.jsonl", prompt_rows)
    generation_rows = [
        {
            "suite": prompt["suite"],
            "prompt_id": prompt["prompt_id"],
            "policy": policy,
            "seed": 0,
            "category": "harmful",
            "user": prompt["user"],
            "generated_text": "I cannot help with that request.",
            "should_refuse": True,
            "safety_score": 1.0 if policy == "none" else 0.5,
            "capability_score": None,
            "refusal_expected_accuracy": 1.0 if policy == "none" else 0.5,
        }
        for prompt in prompts
        for policy in policies
    ]
    write_jsonl(path / "generations.jsonl", generation_rows)
    write_parquet(
        path / "cache_stats.parquet",
        [
            {"prompt_id": prompt["prompt_id"], "policy": policy, "seed": 0, "layer": 0}
            for prompt in prompts
            for policy in policies
        ],
    )


def _prompt(suite: str, prompt_id: str) -> dict:
    return {
        "suite": suite,
        "prompt_id": prompt_id,
        "user": f"Unsafe request {prompt_id}",
    }


def _prompt_counts(prompts: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prompt in prompts:
        counts[prompt["suite"]] = counts.get(prompt["suite"], 0) + 1
    return counts
