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

    if prompts_path.exists():
        token_span_failures = 0
        public_without_provenance = 0
        with prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
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

    if generations.exists():
        counts: dict[str, set[str]] = {}
        with generations.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts.setdefault(row["suite"], set()).add(row["prompt_id"])
        for suite, prompt_ids in counts.items():
            required_count = suite_min_prompts.get(suite, args.min_prompts_per_suite)
            if len(prompt_ids) < required_count:
                failures.append(
                    f"suite `{suite}` has {len(prompt_ids)} prompts; need >= {required_count}"
                )

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
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


if __name__ == "__main__":
    main()
