from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.cache_policies.registry import build_cache_policy, cache_policy_label
from cache_safety_erasure.config import dump_yaml, parse_experiment_config
from cache_safety_erasure.evals.io import load_prompt_suite, load_prompt_suite_manifest
from cache_safety_erasure.evals.rendering import rendered_prompt_manifest
from cache_safety_erasure.evals.spans import character_span_manifest
from cache_safety_erasure.generation.runner import generate_one
from cache_safety_erasure.metrics.aggregate import compute_example_metrics, compute_run_metrics
from cache_safety_erasure.models.loader import hf_device_map, load_model
from cache_safety_erasure.utils.io import (
    append_jsonl,
    environment_snapshot,
    make_run_dir,
    read_jsonl,
    utc_timestamp,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache safety erasure experiments.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", help="Override run.run_id without editing the config file.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the target run directory and skip completed prompt/policy/seed rows.",
    )
    args = parser.parse_args()

    config, raw_config = parse_experiment_config(args.config)
    if args.run_id or args.resume:
        config = replace(
            config,
            run=replace(
                config.run,
                run_id=args.run_id or config.run.run_id,
                resume=config.run.resume or args.resume,
            ),
        )
        raw_config = _raw_config_with_run_overrides(raw_config, args.run_id, args.resume)
    run_dir = make_run_dir(
        config.run.output_dir, config.run.name, config.run.run_id, config.run.resume
    )
    dump_yaml(raw_config, run_dir / "config.resolved.yaml")
    env = environment_snapshot()
    write_json(run_dir / "environment.json", env)
    generations_path = run_dir / "generations.jsonl"
    existing = read_jsonl(generations_path) if config.run.resume else []
    if config.run.resume:
        existing = _reconcile_resume_generations(run_dir, existing)
    done_keys = {
        (row["prompt_id"], row["suite"], row["policy"], int(row["seed"])) for row in existing
    }

    model_bundle = load_model(config.model)
    device_map = (
        hf_device_map(model_bundle.model)
        if getattr(model_bundle, "model", None) is not None
        else None
    )
    suite_prompts = {}
    prompt_counts = {}
    prompt_suite_manifests = {}
    for suite in config.prompt_suites:
        prompts = load_prompt_suite(suite)
        if config.limit_per_suite is not None:
            prompts = prompts[: config.limit_per_suite]
        suite_prompts[suite] = prompts
        prompt_counts[suite] = len(prompts)
        prompt_suite_manifests[suite] = load_prompt_suite_manifest(suite)
    policy_labels = [cache_policy_label(policy) for policy in config.cache_policies]
    expected_generation_count = sum(prompt_counts.values()) * len(config.seeds) * len(policy_labels)

    write_json(
        run_dir / "manifest.json",
        {
            "run_name": config.run.name,
            "model_id": config.model.model_id,
            "model_provider": config.model.provider,
            "model_config": asdict(config.model),
            "model_device_map": device_map,
            "git_commit": env.get("git_commit"),
            "git_dirty": env.get("git_dirty"),
            "config_sha256": _stable_hash(raw_config),
            "prompt_suites": list(config.prompt_suites),
            "prompt_counts": prompt_counts,
            "prompt_suite_manifests": prompt_suite_manifests,
            "cache_policy_configs": [_policy_manifest(policy) for policy in config.cache_policies],
            "cache_policy_labels": policy_labels,
            "seeds": list(config.seeds),
            "limit_per_suite": config.limit_per_suite,
            "expected_generation_count": expected_generation_count,
        },
    )

    cache_stat_rows: list[dict] = []
    cache_stats_sink = _CacheStatsSink(run_dir / "cache_stats.parquet", resume=config.run.resume)
    prompt_manifest_rows: list[dict] = []

    for _suite, prompts in suite_prompts.items():
        prompt_manifest_rows.extend(
            {
                "prompt_id": prompt.id,
                "suite": prompt.suite,
                "category": prompt.category,
                "system": prompt.system,
                "user": prompt.user,
                "should_refuse": prompt.should_refuse,
                "expected_answer": prompt.expected_answer,
                "choices": prompt.choices,
                "hidden_system": prompt.hidden_system,
                "prompt_sha256": _stable_hash(prompt.to_dict()),
                "character_spans": character_span_manifest(prompt),
                "rendered_prompt": rendered_prompt_manifest(
                    getattr(model_bundle, "tokenizer", None), prompt
                ),
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
                        "system": prompt.system,
                        "user": prompt.user,
                        "should_refuse": prompt.should_refuse,
                        "expected_answer": prompt.expected_answer,
                        "hidden_system": prompt.hidden_system,
                        "prompt_metadata": prompt.metadata,
                        "generated_text": result.text,
                        **metrics,
                    }
                    append_jsonl(generations_path, [row])
                    for decision in result.cache_decisions:
                        cache_stat_rows.extend(decision.to_rows(prompt.id, seed))
                        if len(cache_stat_rows) >= 50_000:
                            cache_stats_sink.write(cache_stat_rows)
                            cache_stat_rows = []

    rows = read_jsonl(generations_path)
    metrics = compute_run_metrics(rows)
    write_jsonl(run_dir / "prompts.jsonl", prompt_manifest_rows)
    write_json(run_dir / "metrics.json", metrics)
    if cache_stat_rows:
        cache_stats_sink.write(cache_stat_rows)
    cache_stats_sink.close()
    print(f"Completed run: {run_dir}")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_config_with_run_overrides(
    raw_config: dict[str, Any], run_id: str | None, resume: bool
) -> dict[str, Any]:
    updated = json.loads(json.dumps(raw_config, default=str))
    run = updated.setdefault("run", {})
    if run_id:
        run["run_id"] = run_id
    if resume:
        run["resume"] = True
    return updated


def _policy_manifest(policy: Any) -> dict[str, Any]:
    return asdict(policy)


def _reconcile_resume_generations(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    cache_keys = _cache_stats_generation_keys(run_dir / "cache_stats.parquet")
    kept = [
        row
        for row in rows
        if (
            str(row.get("prompt_id")),
            str(row.get("policy")),
            int(row.get("seed", 0)),
        )
        in cache_keys
    ]
    if len(kept) == len(rows):
        return rows
    orphan_path = run_dir / f"generations.orphaned_without_cache_stats.{utc_timestamp()}.jsonl"
    write_jsonl(orphan_path, rows)
    write_jsonl(run_dir / "generations.jsonl", kept)
    print(
        "Resume reconciliation removed "
        f"{len(rows) - len(kept)} generation row(s) without cache-stat evidence; "
        f"archived original rows at {orphan_path}."
    )
    return kept


def _cache_stats_generation_keys(cache_stats_path: Path) -> set[tuple[str, str, int]]:
    if not cache_stats_path.exists():
        return set()
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for cache stats resume reconciliation.") from exc
    try:
        table = pq.read_table(cache_stats_path, columns=["prompt_id", "policy", "seed"])
    except Exception:
        return set()
    prompt_ids = table.column("prompt_id").to_pylist()
    policies = table.column("policy").to_pylist()
    seeds = table.column("seed").to_pylist()
    return {
        (str(prompt_id), str(policy), int(seed or 0))
        for prompt_id, policy, seed in zip(prompt_ids, policies, seeds, strict=False)
        if prompt_id is not None and policy is not None
    }


CACHE_STATS_COLUMNS = [
    "prompt_id",
    "seed",
    "policy",
    "decode_step",
    "original_seq_len",
    "retained_count",
    "evicted_count",
    "retained_indices",
    "evicted_indices",
    "layer_count",
    "cache_l2_before",
    "cache_l2_after",
    "retained_special_tokens",
    "retained_template_tokens",
    "retained_system_tokens",
    "retained_user_tokens",
    "retained_generated_tokens",
    "retained_unknown_tokens",
    "evicted_special_tokens",
    "evicted_template_tokens",
    "evicted_system_tokens",
    "evicted_user_tokens",
    "evicted_generated_tokens",
    "evicted_unknown_tokens",
    "sink_tokens",
    "recent_tokens",
    "policy_seed",
    "attention_scores_used",
    "quantization_bits",
    "protected_spans",
    "protected_candidate_count",
    "protected_retained_count",
    "protected_dropped_count",
    "patched_from_baseline",
    "patched_token_count",
    "patched_roles",
    "patched_token_indices",
    "patch_selection",
    "patch_matched_roles",
    "patch_layers",
    "patch_heads",
    "patch_components",
    "cache_l2_after_patch",
]


def _normalize_cache_stat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in CACHE_STATS_COLUMNS} for row in rows]


def _cache_stats_schema() -> Any:
    import pyarrow as pa

    int_columns = {
        "seed",
        "decode_step",
        "original_seq_len",
        "retained_count",
        "evicted_count",
        "layer_count",
        "retained_special_tokens",
        "retained_template_tokens",
        "retained_system_tokens",
        "retained_user_tokens",
        "retained_generated_tokens",
        "retained_unknown_tokens",
        "evicted_special_tokens",
        "evicted_template_tokens",
        "evicted_system_tokens",
        "evicted_user_tokens",
        "evicted_generated_tokens",
        "evicted_unknown_tokens",
        "sink_tokens",
        "recent_tokens",
        "policy_seed",
        "quantization_bits",
        "protected_candidate_count",
        "protected_retained_count",
        "protected_dropped_count",
        "patched_token_count",
    }
    float_columns = {"cache_l2_before", "cache_l2_after", "cache_l2_after_patch"}
    bool_columns = {"attention_scores_used", "patched_from_baseline"}
    fields = []
    for column in CACHE_STATS_COLUMNS:
        if column in int_columns:
            fields.append(pa.field(column, pa.int64()))
        elif column in float_columns:
            fields.append(pa.field(column, pa.float64()))
        elif column in bool_columns:
            fields.append(pa.field(column, pa.bool_()))
        else:
            fields.append(pa.field(column, pa.large_string()))
    return pa.schema(fields)


class _CacheStatsSink:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.resume = resume
        self.writer: Any | None = None
        self.temp_path: Path | None = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        schema = _cache_stats_schema()
        table = _align_table_to_schema(_cache_stats_table(rows), schema)
        if self.writer is None:
            write_path = self.path
            if self.resume and self.path.exists():
                self.temp_path = self.path.with_suffix(".parquet.tmp")
                write_path = self.temp_path
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
            self.writer = pq.ParquetWriter(write_path, schema)
            if self.temp_path is not None:
                _copy_existing_cache_stats(self.path, self.writer, schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            if self.temp_path is not None:
                self.temp_path.replace(self.path)
            return
        if not self.path.exists():
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
            pq.write_table(_cache_stats_table([]), self.path)


def _cache_stats_table(rows: list[dict[str, Any]]) -> Any:
    try:
        import pyarrow as pa
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
    normalized = _normalize_cache_stat_rows(rows)
    schema = _cache_stats_schema()
    arrays = [
        pa.array([row[field.name] for row in normalized], type=field.type) for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _copy_existing_cache_stats(path: Path, writer: Any, schema: Any) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for resume-safe cache stats.") from exc
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        writer.write_table(_align_table_to_schema(table, schema))


def _align_table_to_schema(table: Any, schema: Any) -> Any:
    import pyarrow as pa

    columns = []
    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]
            if not column.type.equals(field.type):
                column = column.cast(field.type)
            columns.append(column)
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


if __name__ == "__main__":
    main()
