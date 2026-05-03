from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.cache_policies.registry import build_cache_policy
from cache_safety_erasure.config import dump_yaml, parse_experiment_config
from cache_safety_erasure.evals.io import load_prompt_suite
from cache_safety_erasure.evals.spans import character_span_manifest
from cache_safety_erasure.generation.runner import generate_one
from cache_safety_erasure.metrics.aggregate import compute_example_metrics, compute_run_metrics
from cache_safety_erasure.models.loader import load_model
from cache_safety_erasure.utils.io import (
    append_jsonl,
    environment_snapshot,
    make_run_dir,
    read_jsonl,
    write_json,
    write_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache safety erasure experiments.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config, raw_config = parse_experiment_config(args.config)
    run_dir = make_run_dir(
        config.run.output_dir, config.run.name, config.run.run_id, config.run.resume
    )
    dump_yaml(raw_config, run_dir / "config.resolved.yaml")
    env = environment_snapshot()
    write_json(run_dir / "environment.json", env)
    write_json(
        run_dir / "manifest.json",
        {
            "run_name": config.run.name,
            "model_id": config.model.model_id,
            "model_provider": config.model.provider,
            "git_commit": env.get("git_commit"),
            "prompt_suites": list(config.prompt_suites),
            "cache_policies": [policy.name for policy in config.cache_policies],
            "seeds": list(config.seeds),
        },
    )

    generations_path = run_dir / "generations.jsonl"
    existing = read_jsonl(generations_path) if config.run.resume else []
    done_keys = {
        (row["prompt_id"], row["suite"], row["policy"], int(row["seed"])) for row in existing
    }

    model_bundle = load_model(config.model)
    cache_stat_rows: list[dict] = []
    prompt_manifest_rows: list[dict] = []

    for suite in config.prompt_suites:
        prompts = load_prompt_suite(suite)
        if config.limit_per_suite is not None:
            prompts = prompts[: config.limit_per_suite]
        prompt_manifest_rows.extend(
            {
                "prompt_id": prompt.id,
                "suite": prompt.suite,
                "category": prompt.category,
                "should_refuse": prompt.should_refuse,
                "expected_answer": prompt.expected_answer,
                "character_spans": character_span_manifest(prompt),
                "metadata": prompt.metadata,
            }
            for prompt in prompts
        )
        for seed in config.seeds:
            for policy_config in config.cache_policies:
                policy = build_cache_policy(policy_config, seed)
                policy_name = getattr(policy, "name", policy_config.name)
                for prompt in prompts:
                    key = (prompt.id, prompt.suite, policy_name, seed)
                    if key in done_keys:
                        continue
                    result = generate_one(
                        model_bundle=model_bundle,
                        prompt=prompt,
                        policy=policy,
                        generation_config=config.generation,
                        patch_from_baseline=policy_config.patch_from_baseline,
                    )
                    metrics = compute_example_metrics(prompt, result.text)
                    row = {
                        "prompt_id": prompt.id,
                        "suite": prompt.suite,
                        "category": prompt.category,
                        "policy": policy_name,
                        "seed": seed,
                        "model_id": config.model.model_id,
                        "user": prompt.user,
                        "generated_text": result.text,
                        **metrics,
                    }
                    append_jsonl(generations_path, [row])
                    for decision in result.cache_decisions:
                        try:
                            layer_count = None
                        except Exception:
                            layer_count = None
                        cache_stat_rows.extend(decision.to_rows(prompt.id, seed, layer_count))

    rows = read_jsonl(generations_path)
    metrics = compute_run_metrics(rows)
    append_jsonl(run_dir / "prompts.jsonl", prompt_manifest_rows)
    write_json(run_dir / "metrics.json", metrics)
    if cache_stat_rows:
        write_parquet(run_dir / "cache_stats.parquet", cache_stat_rows)
    else:
        write_parquet(run_dir / "cache_stats.parquet", [])
    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
