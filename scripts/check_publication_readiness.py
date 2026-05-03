from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a result directory is paper-ready.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--min-prompts-per-suite", type=int, default=100)
    parser.add_argument(
        "--suite-min-prompts",
        action="append",
        default=[],
        help="Optional per-suite threshold override, e.g. system_leakage=2.",
    )
    parser.add_argument("--max-ci-width", type=float, default=0.08)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-mock-model", action="store_true")
    parser.add_argument("--allow-tiny-model", action="store_true")
    parser.add_argument("--allow-smoke-run", action="store_true")
    parser.add_argument("--min-policies", type=int, default=3)
    parser.add_argument("--required-suite", action="append", default=[])
    parser.add_argument("--required-policy", action="append", default=[])
    parser.add_argument("--require-public-provenance", action="store_true")
    parser.add_argument("--require-causal-patch", action="store_true")
    parser.add_argument("--require-policy-pinned", action="store_true")
    parser.add_argument(
        "--allow-inactive-compression",
        action="store_true",
        help="Allow policies whose cache stats show no eviction or quantization activity.",
    )
    args = parser.parse_args()
    suite_min_prompts = _parse_suite_min_prompts(args.suite_min_prompts)

    failures: list[str] = []
    generations = args.results_dir / "generations.jsonl"
    metrics_path = args.results_dir / "metrics.json"
    manifest_path = args.results_dir / "manifest.json"
    prompts_path = args.results_dir / "prompts.jsonl"
    for required in [
        "config.resolved.yaml",
        "environment.json",
        "manifest.json",
        "prompts.jsonl",
        "generations.jsonl",
        "metrics.json",
        "cache_stats.parquet",
    ]:
        if not (args.results_dir / required).exists():
            failures.append(f"missing artifact: {required}")

    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("git_dirty") and not args.allow_dirty:
            failures.append("run was produced from a dirty git working tree")
        if manifest.get("model_provider") == "mock" and not args.allow_mock_model:
            failures.append("mock model runs are not paper evidence")
        model_id = str(manifest.get("model_id", ""))
        if "tiny" in model_id.lower() and not args.allow_tiny_model:
            failures.append(f"tiny model `{model_id}` is not paper evidence")
        run_name = str(manifest.get("run_name", ""))
        if "smoke" in run_name.lower() and not args.allow_smoke_run:
            failures.append(f"smoke run `{run_name}` is not paper evidence")
        if not manifest.get("cache_policy_configs"):
            failures.append("manifest lacks full cache policy configs")
        if not manifest.get("cache_policy_labels"):
            failures.append("manifest lacks cache policy labels")
        if manifest.get("expected_generation_count") is None:
            failures.append("manifest lacks expected generation count")
        if not manifest.get("prompt_counts"):
            failures.append("manifest lacks prompt counts")
        policy_configs = manifest.get("cache_policy_configs") or []
        if len(policy_configs) < args.min_policies:
            failures.append(f"manifest has {len(policy_configs)} policies; need >= {args.min_policies}")
        policy_names = {str(policy.get("name")) for policy in policy_configs if isinstance(policy, dict)}
        for required_policy in args.required_policy:
            if required_policy not in policy_names and not any(
                str(policy.get("name", "")).startswith(required_policy)
                for policy in policy_configs
                if isinstance(policy, dict)
            ):
                failures.append(f"missing required policy `{required_policy}`")
        if args.require_policy_pinned and "policy_pinned" not in policy_names:
            failures.append("missing policy_pinned mitigation policy")
        if args.require_causal_patch and not any(
            isinstance(policy, dict) and policy.get("patch_from_baseline")
            for policy in policy_configs
        ):
            failures.append("missing cache patch policy with patch_from_baseline")
        prompt_counts = manifest.get("prompt_counts") or {}
        for required_suite in args.required_suite:
            if required_suite not in prompt_counts:
                failures.append(f"missing required suite `{required_suite}`")

    prompt_rows: list[dict] = []
    if prompts_path.exists():
        token_span_failures = 0
        public_without_provenance = 0
        with prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_rows.append(row)
                rendered = row.get("rendered_prompt", {})
                if manifest.get("model_provider") != "mock" and (
                    rendered.get("token_count") is None or not rendered.get("token_role_spans")
                ):
                    token_span_failures += 1
                if args.require_public_provenance and str(row.get("suite", "")).startswith("public_"):
                    metadata = row.get("metadata", {})
                    if not metadata.get("source_dataset") or not metadata.get("source_split"):
                        public_without_provenance += 1
        if token_span_failures:
            failures.append(f"{token_span_failures} prompts lack tokenizer token-role spans")
        if public_without_provenance:
            failures.append(f"{public_without_provenance} public prompts lack dataset provenance")

    figure_dir = args.results_dir / "figures"
    if not figure_dir.exists() or not list(figure_dir.glob("*.png")):
        failures.append("missing generated PNG figures")
    if not args.allow_inactive_compression and (args.results_dir / "cache_stats.parquet").exists():
        _check_active_compression(args.results_dir / "cache_stats.parquet", manifest, failures)

    generation_rows: list[dict] = []
    if generations.exists():
        counts: dict[str, set[str]] = {}
        with generations.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                generation_rows.append(row)
                counts.setdefault(row["suite"], set()).add(row["prompt_id"])
        for suite, prompt_ids in counts.items():
            required_count = suite_min_prompts.get(suite, args.min_prompts_per_suite)
            if len(prompt_ids) < required_count:
                failures.append(
                    f"suite `{suite}` has {len(prompt_ids)} prompts; need >= {required_count}"
                )
        if manifest:
            _check_generation_matrix(manifest, prompt_rows, generation_rows, failures)

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        policy_summary = metrics.get("publication_summary", {}).get("policies", {})
        has_global_safety = any(
            value.get("mean_safety_score") is not None for value in policy_summary.values()
        )
        has_global_capability = any(
            value.get("mean_capability_score") is not None for value in policy_summary.values()
        )
        if has_global_safety and has_global_capability:
            contrasts = metrics.get("policy_level_contrasts", {})
            if not contrasts:
                failures.append("missing policy-level safety-vs-capability contrasts")
            for policy, contrast in contrasts.items():
                ssei_ci = contrast.get("selective_safety_erasure_index_ci", {})
                if policy != "none" and ssei_ci.get("mean") is None:
                    failures.append(f"{policy}: missing policy-level SSEI CI")
        if args.require_causal_patch and not metrics.get("causal_restoration"):
            failures.append("missing causal restoration metrics")
        for key, value in metrics.get("selective_safety_erasure", {}).items():
            ci = value.get("paired_safety_degradation_ci", {})
            if ci.get("ci_low") is None or ci.get("ci_high") is None:
                failures.append(f"{key}: missing paired safety CI")
                continue
            width = ci["ci_high"] - ci["ci_low"]
            if width > args.max_ci_width:
                failures.append(
                    f"{key}: paired safety CI width {width:.3f}; target <= {args.max_ci_width:.3f}"
                )

    if failures:
        print("NOT PAPER READY")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PAPER READY CHECK PASSED")


def _parse_suite_min_prompts(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Expected --suite-min-prompts value like suite=100, got `{value}`")
        suite, raw_count = value.split("=", 1)
        parsed[suite] = int(raw_count)
    return parsed


def _check_generation_matrix(
    manifest: dict,
    prompt_rows: list[dict],
    generation_rows: list[dict],
    failures: list[str],
) -> None:
    expected_count = manifest.get("expected_generation_count")
    if expected_count is not None and len(generation_rows) != int(expected_count):
        failures.append(
            f"generation row count is {len(generation_rows)}; expected {int(expected_count)}"
        )

    policy_labels = [str(label) for label in manifest.get("cache_policy_labels", [])]
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if not prompt_rows or not policy_labels or not seeds:
        return

    expected_keys = {
        (str(prompt["suite"]), str(prompt["prompt_id"]), policy, seed)
        for prompt in prompt_rows
        for policy in policy_labels
        for seed in seeds
    }
    observed_keys = []
    malformed = 0
    for row in generation_rows:
        try:
            seed = int(row["seed"])
            observed_keys.append(
                (str(row["suite"]), str(row["prompt_id"]), str(row["policy"]), seed)
            )
        except (KeyError, TypeError, ValueError):
            malformed += 1
    if malformed:
        failures.append(f"{malformed} generation rows have malformed matrix keys")
    observed_key_set = set(observed_keys)
    duplicate_count = len(observed_keys) - len(observed_key_set)
    if duplicate_count:
        failures.append(f"generation matrix has {duplicate_count} duplicate rows")
    missing = expected_keys - observed_key_set
    extra = observed_key_set - expected_keys
    if missing:
        failures.append(
            f"generation matrix is missing {len(missing)} rows; "
            f"first missing: {_format_matrix_key(sorted(missing)[0])}"
        )
    if extra:
        failures.append(
            f"generation matrix has {len(extra)} rows outside the manifest; "
            f"first extra: {_format_matrix_key(sorted(extra)[0])}"
        )


def _format_matrix_key(key: tuple[str, str, str, int]) -> str:
    suite, prompt_id, policy, seed = key
    return f"suite={suite}, prompt_id={prompt_id}, policy={policy}, seed={seed}"


def _check_active_compression(cache_stats_path: Path, manifest: dict, failures: list[str]) -> None:
    expected_policies = [
        str(policy)
        for policy in manifest.get("cache_policy_labels", [])
        if str(policy) != "none"
    ]
    if not expected_policies:
        return
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        failures.append("pyarrow is required to validate active compression")
        return
    try:
        parquet_file = pq.ParquetFile(cache_stats_path)
    except Exception as exc:
        failures.append(f"cannot inspect cache stats for active compression: {exc}")
        return
    available_columns = set(parquet_file.schema.names)
    required_columns = {
        "policy",
        "original_seq_len",
        "evicted_count",
        "quantization_bits",
        "cache_l2_before",
        "cache_l2_after",
    }
    columns = [column for column in required_columns if column in available_columns]
    if "policy" not in columns:
        failures.append("cache stats lack policy column for active-compression check")
        return
    stats: dict[str, dict[str, float]] = {
        policy: {"rows": 0.0, "evicted": 0.0, "quantized": 0.0, "l2_delta": 0.0}
        for policy in expected_policies
    }
    for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
        table = batch.to_pydict()
        policies = table.get("policy", [])
        for idx, raw_policy in enumerate(policies):
            policy = str(raw_policy)
            if policy not in stats:
                continue
            stats[policy]["rows"] += 1
            original_seq_len = _float_at(table, "original_seq_len", idx)
            evicted_count = _float_at(table, "evicted_count", idx)
            quantization_bits = table.get("quantization_bits", [None] * len(policies))[idx]
            before = _float_at(table, "cache_l2_before", idx)
            after = _float_at(table, "cache_l2_after", idx)
            stats[policy]["evicted"] += evicted_count
            if quantization_bits is not None and original_seq_len > 0:
                stats[policy]["quantized"] += 1
            stats[policy]["l2_delta"] += abs(before - after)
    for policy, policy_stats in stats.items():
        if policy_stats["rows"] == 0:
            failures.append(f"cache policy `{policy}` has no cache-stat rows")
            continue
        if policy_stats["evicted"] <= 0 and policy_stats["quantized"] <= 0:
            failures.append(
                f"cache policy `{policy}` appears inactive: no evictions or quantization rows"
            )


def _float_at(table: dict[str, list], column: str, idx: int) -> float:
    values = table.get(column)
    if values is None:
        return 0.0
    value = values[idx]
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
