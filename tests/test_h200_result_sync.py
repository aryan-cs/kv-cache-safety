import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from check_h200_fetch_manifest import check_h200_fetch_manifest
from compare_artifact_manifests import compare_manifest_files, compare_manifests
from write_artifact_manifest import artifact_manifest


def test_artifact_manifest_hashes_requested_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "results" / "run").mkdir(parents=True)
    generation = tmp_path / "results" / "run" / "generations.jsonl"
    metrics = tmp_path / "results" / "run" / "metrics.json"
    generation.write_text("{}\n", encoding="utf-8")
    metrics.write_text('{"ok": true}\n', encoding="utf-8")

    manifest = artifact_manifest(tmp_path, [Path("results/run")])

    assert manifest["missing_paths"] == []
    assert manifest["file_count"] == 2
    rows = {row["path"]: row for row in manifest["files"]}
    assert rows["results/run/generations.jsonl"]["sha256"] == hashlib.sha256(
        generation.read_bytes()
    ).hexdigest()
    assert rows["results/run/metrics.json"]["bytes"] == metrics.stat().st_size


def test_artifact_manifest_rejects_paths_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        artifact_manifest(tmp_path, [Path("../outside")])


def test_compare_artifact_manifests_reports_hash_mismatch() -> None:
    expected = {
        "files": [{"path": "results/run/metrics.json", "bytes": 3, "sha256": "abc"}],
        "missing_paths": [],
    }
    actual = {
        "files": [{"path": "results/run/metrics.json", "bytes": 3, "sha256": "def"}],
        "missing_paths": [],
    }

    report = compare_manifests(expected, actual)

    assert report["passed"] is False
    assert report["failures"] == ["sha256_mismatch:results/run/metrics.json"]


def test_fetch_h200_results_script_is_guarded_and_checksum_verified() -> None:
    script = Path("scripts/fetch_h200_results.sh").read_text(encoding="utf-8")

    assert "/home/aryang9/sandbox/llm-safety" in script
    assert "safe_artifact_path" in script
    assert "results/*|paper/generated/*|paper/audit/*" in script
    default_block = script.split("remote_generated_paths=", maxsplit=1)[0]
    assert "paper/generated/h200_qwen_full_sweep" not in default_block
    assert "h200_attention_diagnostic_qwen7b" not in default_block
    assert "FETCH_H200_REMOTE_GENERATED" in script
    assert "FETCH_H200_FINALIZED" in script
    assert "finalized_optional_paths" in script
    assert "h200_qwen_full_sweep_plus_ci_extension" in script
    assert "paper/generated/claim_assessment" in script
    assert "paper/audit/h200_qwen_full_sweep_plus_ci_extension_summary" in script
    assert "paper/audit/h200_causal_patch_qwen7b_summary" in script
    assert "h200_causal_patch_qwen7b_audit_blinded_annotator_open_judge_v3.csv" in script
    assert "Skipping absent optional remote artifact path" in script
    assert "paper/generated/preliminary_claim_assessment" in script
    assert "h200_qwen_full_sweep_audit_key.jsonl" in script
    assert "h200_qwen_full_sweep_audit_export_manifest.json" in script
    assert "h200_qwen_full_sweep_audit_blinded_annotator_01.csv" in script
    assert "h200_qwen_full_sweep_audit_blinded_annotator_02.csv" in script
    assert "h200_causal_patch_qwen7b_audit_export_manifest.json" in script
    assert "h200_causal_patch_qwen7b_audit_blinded_annotator_01.csv" in script
    assert "h200_causal_patch_qwen7b_audit_blinded_annotator_02.csv" in script
    assert "rsync -az --checksum" in script
    assert "fetch_with_tar" in script
    assert "tar -cf -" in script
    assert "scripts/write_artifact_manifest.py" in script
    assert "scripts/compare_artifact_manifests.py" in script
    assert "git pull" not in script


def test_compare_artifact_manifests_reports_missing_paths() -> None:
    report = compare_manifests({"files": [], "missing_paths": ["results/run"]}, {"files": []})

    assert report["passed"] is False
    assert report["failures"] == ["missing_remote_path:results/run"]


def test_h200_fetch_manifest_check_requires_fetched_primary_and_causal_raw_files(
    tmp_path: Path,
) -> None:
    required_paths = ["results/h200_qwen_full_sweep", "results/h200_causal_patch_qwen7b"]
    for required_path in required_paths:
        run_dir = tmp_path / required_path
        run_dir.mkdir(parents=True)
        for filename in [
            "config.resolved.yaml",
            "environment.json",
            "manifest.json",
            "prompts.jsonl",
            "generations.jsonl",
            "cache_stats.parquet",
        ]:
            (run_dir / filename).write_text(f"{required_path}/{filename}\n", encoding="utf-8")
    local_manifest_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_local.json"
    remote_manifest_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_remote.json"
    compare_report_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_compare.json"
    local_manifest_path.parent.mkdir(parents=True)
    manifest_text = json_dump(artifact_manifest(tmp_path, [Path(path) for path in required_paths]))
    local_manifest_path.write_text(manifest_text, encoding="utf-8")
    remote_manifest_path.write_text(manifest_text, encoding="utf-8")
    compare_report_path.write_text(
        json_dump(compare_manifest_files(remote_manifest_path, local_manifest_path)),
        encoding="utf-8",
    )

    report = check_h200_fetch_manifest(
        root=tmp_path,
        remote_manifest_path=remote_manifest_path,
        local_manifest_path=local_manifest_path,
        compare_report_path=compare_report_path,
        required_paths=required_paths,
    )

    assert report["passed"] is True
    assert report["failures"] == []


def test_h200_fetch_manifest_check_rejects_stale_or_incomplete_fetch(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "h200_qwen_full_sweep"
    run_dir.mkdir(parents=True)
    for filename in [
        "config.resolved.yaml",
        "environment.json",
        "manifest.json",
        "prompts.jsonl",
        "generations.jsonl",
        "cache_stats.parquet",
    ]:
        (run_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    local_manifest_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_local.json"
    remote_manifest_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_remote.json"
    compare_report_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_compare.json"
    local_manifest_path.parent.mkdir(parents=True)
    local_manifest = artifact_manifest(tmp_path, [Path("results/h200_qwen_full_sweep")])
    local_manifest_path.write_text(json_dump(local_manifest), encoding="utf-8")
    remote_manifest_path.write_text(json_dump(local_manifest), encoding="utf-8")
    compare_report_path.write_text(
        json_dump(compare_manifest_files(remote_manifest_path, local_manifest_path)),
        encoding="utf-8",
    )
    (run_dir / "generations.jsonl").write_text("stale local edit\n", encoding="utf-8")

    report = check_h200_fetch_manifest(
        root=tmp_path,
        remote_manifest_path=remote_manifest_path,
        local_manifest_path=local_manifest_path,
        compare_report_path=compare_report_path,
        required_paths=[
            "results/h200_qwen_full_sweep",
            "results/h200_causal_patch_qwen7b",
        ],
    )

    assert report["passed"] is False
    assert "required_raw_file_sha256_mismatch:results/h200_qwen_full_sweep/generations.jsonl" in report[
        "failures"
    ]
    assert "missing_requested_path:results/h200_causal_patch_qwen7b" in report["failures"]
    assert (
        "manifest_lacks_required_raw_file:"
        "results/h200_causal_patch_qwen7b/generations.jsonl"
    ) in report["failures"]


def test_h200_fetch_manifest_check_rejects_stale_compare_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "h200_qwen_full_sweep"
    run_dir.mkdir(parents=True)
    for filename in [
        "config.resolved.yaml",
        "environment.json",
        "manifest.json",
        "prompts.jsonl",
        "generations.jsonl",
        "cache_stats.parquet",
    ]:
        (run_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    local_manifest_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_local.json"
    remote_manifest_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_remote.json"
    compare_report_path = tmp_path / "logs" / "h200" / "h200_artifact_manifest_compare.json"
    local_manifest_path.parent.mkdir(parents=True)
    manifest_text = json_dump(artifact_manifest(tmp_path, [Path("results/h200_qwen_full_sweep")]))
    local_manifest_path.write_text(manifest_text, encoding="utf-8")
    remote_manifest_path.write_text(manifest_text, encoding="utf-8")
    compare_report_path.write_text(
        json_dump(compare_manifest_files(remote_manifest_path, local_manifest_path)),
        encoding="utf-8",
    )
    local_manifest_path.write_text(
        json_dump(artifact_manifest(tmp_path, [Path("results/h200_qwen_full_sweep")])),
        encoding="utf-8",
    )
    local_manifest_path.write_text(local_manifest_path.read_text(encoding="utf-8") + "\n")

    report = check_h200_fetch_manifest(
        root=tmp_path,
        remote_manifest_path=remote_manifest_path,
        local_manifest_path=local_manifest_path,
        compare_report_path=compare_report_path,
        required_paths=["results/h200_qwen_full_sweep"],
    )

    assert report["passed"] is False
    assert "artifact_manifest_compare_actual_manifest_sha256_stale" in report["failures"]


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, indent=2) + "\n"
