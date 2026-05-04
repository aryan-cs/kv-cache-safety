from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.metrics.aggregate import compute_run_metrics
from cache_safety_erasure.utils.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a disjoint CI-extension run into the primary result matrix. "
            "Duplicate prompt clusters are skipped, so the merged directory only adds "
            "new prompt-cluster evidence."
        )
    )
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--extension-results-dir", type=Path, required=True)
    parser.add_argument("--output-results-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merge_ci_extension_results(
        primary_results_dir=args.primary_results_dir,
        extension_results_dir=args.extension_results_dir,
        output_results_dir=args.output_results_dir,
        overwrite=args.overwrite,
    )
    print(f"Wrote merged result directory: {args.output_results_dir}")


def merge_ci_extension_results(
    *,
    primary_results_dir: Path,
    extension_results_dir: Path,
    output_results_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    _require_run_artifacts(primary_results_dir)
    _require_run_artifacts(extension_results_dir)
    if output_results_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {output_results_dir}")
        shutil.rmtree(output_results_dir)
    output_results_dir.mkdir(parents=True)
    (output_results_dir / "figures").mkdir()

    primary_manifest = _read_json(primary_results_dir / "manifest.json")
    extension_manifest = _read_json(extension_results_dir / "manifest.json")
    _check_compatible_manifests(primary_manifest, extension_manifest)

    primary_prompts = read_jsonl(primary_results_dir / "prompts.jsonl")
    extension_prompts = read_jsonl(extension_results_dir / "prompts.jsonl")
    primary_generations = read_jsonl(primary_results_dir / "generations.jsonl")
    extension_generations = read_jsonl(extension_results_dir / "generations.jsonl")

    primary_prompt_keys = {_prompt_key(row) for row in primary_prompts}
    extension_prompt_rows = [
        row for row in extension_prompts if _prompt_key(row) not in primary_prompt_keys
    ]
    added_prompt_keys = {_prompt_key(row) for row in extension_prompt_rows}
    skipped_duplicate_prompt_count = len(extension_prompts) - len(extension_prompt_rows)
    extension_generation_rows = [
        row for row in extension_generations if _prompt_key(row) in added_prompt_keys
    ]
    _check_extension_matrix(
        extension_prompt_rows,
        extension_generation_rows,
        policy_labels=[str(label) for label in primary_manifest.get("cache_policy_labels", [])],
        seeds=[int(seed) for seed in primary_manifest.get("seeds", [])],
    )

    combined_prompts = [*primary_prompts, *extension_prompt_rows]
    combined_generations = [*primary_generations, *extension_generation_rows]
    _check_no_duplicate_generation_keys(combined_generations)

    write_jsonl(output_results_dir / "prompts.jsonl", combined_prompts)
    write_jsonl(output_results_dir / "generations.jsonl", combined_generations)
    write_json(output_results_dir / "metrics.json", compute_run_metrics(combined_generations))
    _write_combined_cache_stats(
        primary_results_dir / "cache_stats.parquet",
        extension_results_dir / "cache_stats.parquet",
        output_results_dir / "cache_stats.parquet",
        added_prompt_ids={prompt_id for _suite, prompt_id in added_prompt_keys},
    )
    shutil.copy2(primary_results_dir / "config.resolved.yaml", output_results_dir / "config.resolved.yaml")
    environment = _read_json(primary_results_dir / "environment.json")
    environment["combined_results"] = _combined_environment_note(
        primary_results_dir, extension_results_dir
    )
    write_json(output_results_dir / "environment.json", environment)

    prompt_counts = Counter(str(row["suite"]) for row in combined_prompts)
    manifest = dict(primary_manifest)
    manifest["prompt_counts"] = dict(sorted(prompt_counts.items()))
    manifest["prompt_suites"] = sorted(prompt_counts)
    manifest["expected_generation_count"] = len(combined_generations)
    manifest["prompt_suite_manifests"] = _combined_suite_manifests(combined_prompts)
    manifest["combined_results"] = {
        "primary_results_dir": str(primary_results_dir),
        "extension_results_dir": str(extension_results_dir),
        "skipped_duplicate_prompt_count": skipped_duplicate_prompt_count,
        "added_prompt_counts": dict(Counter(str(row["suite"]) for row in extension_prompt_rows)),
        "generated_by_git_commit": _git_commit(),
    }
    write_json(output_results_dir / "manifest.json", manifest)
    merge_summary = {
        "primary_results_dir": str(primary_results_dir),
        "extension_results_dir": str(extension_results_dir),
        "output_results_dir": str(output_results_dir),
        "primary_prompt_count": len(primary_prompts),
        "extension_prompt_count": len(extension_prompts),
        "added_prompt_count": len(extension_prompt_rows),
        "skipped_duplicate_prompt_count": skipped_duplicate_prompt_count,
        "combined_generation_count": len(combined_generations),
    }
    write_json(output_results_dir / "merge_summary.json", merge_summary)
    return merge_summary


def _require_run_artifacts(results_dir: Path) -> None:
    for required in [
        "manifest.json",
        "environment.json",
        "config.resolved.yaml",
        "prompts.jsonl",
        "generations.jsonl",
        "cache_stats.parquet",
    ]:
        if not (results_dir / required).exists():
            raise SystemExit(f"Missing required artifact: {results_dir / required}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_compatible_manifests(primary: dict[str, Any], extension: dict[str, Any]) -> None:
    checks = {
        "model_id": (primary.get("model_id"), extension.get("model_id")),
        "model_provider": (primary.get("model_provider"), extension.get("model_provider")),
        "cache_policy_labels": (
            primary.get("cache_policy_labels"),
            extension.get("cache_policy_labels"),
        ),
        "seeds": (primary.get("seeds"), extension.get("seeds")),
    }
    failures = [
        f"{name}: primary={left!r}, extension={right!r}"
        for name, (left, right) in checks.items()
        if left != right
    ]
    if failures:
        raise SystemExit("Cannot merge incompatible runs:\n- " + "\n- ".join(failures))


def _prompt_key(row: dict[str, Any]) -> tuple[str, str]:
    prompt_id = row.get("prompt_id", row.get("id"))
    return str(row["suite"]), str(prompt_id)


def _generation_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    suite, prompt_id = _prompt_key(row)
    return suite, prompt_id, str(row["policy"]), int(row["seed"])


def _check_extension_matrix(
    prompt_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    *,
    policy_labels: list[str],
    seeds: list[int],
) -> None:
    expected = {
        (str(prompt["suite"]), str(prompt["prompt_id"]), policy, seed)
        for prompt in prompt_rows
        for policy in policy_labels
        for seed in seeds
    }
    observed = {_generation_key(row) for row in generation_rows}
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={sorted(missing)[0]}")
        if extra:
            details.append(f"extra={len(extra)} first={sorted(extra)[0]}")
        raise SystemExit("Extension run is not a complete matrix for added prompts: " + "; ".join(details))


def _check_no_duplicate_generation_keys(rows: list[dict[str, Any]]) -> None:
    keys = [_generation_key(row) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise SystemExit(f"Merged generations would contain {duplicate_count} duplicate rows.")


def _write_combined_cache_stats(
    primary_path: Path,
    extension_path: Path,
    output_path: Path,
    *,
    added_prompt_ids: set[str],
) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    primary = pq.read_table(primary_path)
    extension = pq.read_table(extension_path)
    if added_prompt_ids:
        mask = pc.is_in(extension["prompt_id"], value_set=pa.array(sorted(added_prompt_ids)))
        extension = extension.filter(mask)
    else:
        extension = extension.slice(0, 0)
    pq.write_table(pa.concat_tables([primary, extension], promote_options="default"), output_path)


def _combined_environment_note(primary_results_dir: Path, extension_results_dir: Path) -> dict[str, Any]:
    return {
        "primary_results_dir": str(primary_results_dir),
        "extension_results_dir": str(extension_results_dir),
        "generated_by_git_commit": _git_commit(),
    }


def _combined_suite_manifests(prompt_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in prompt_rows:
        rows_by_suite.setdefault(str(row["suite"]), []).append(row)
    manifests = {}
    for suite, rows in sorted(rows_by_suite.items()):
        digest = hashlib.sha256()
        prompt_ids = []
        source_datasets = set()
        for row in sorted(rows, key=lambda item: str(item["prompt_id"])):
            prompt_ids.append(str(row["prompt_id"]))
            metadata = row.get("metadata") or {}
            if metadata.get("source_dataset"):
                source_datasets.add(str(metadata["source_dataset"]))
            digest.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
            digest.update(b"\n")
        manifests[suite] = {
            "suite_name": suite,
            "record_count": len(rows),
            "sha256": digest.hexdigest(),
            "source": "merged_result_prompts",
            "source_datasets": sorted(source_datasets),
            "prompt_ids": prompt_ids,
        }
    return manifests


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
