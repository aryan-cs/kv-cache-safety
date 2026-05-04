import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from compare_artifact_manifests import compare_manifests
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
    assert "paper/generated/preliminary_claim_assessment" in script
    assert "paper/generated/claim_assessment" not in script
    assert "h200_qwen_full_sweep_audit_key.jsonl" in script
    assert "h200_qwen_full_sweep_audit_export_manifest.json" in script
    assert "h200_qwen_full_sweep_audit_blinded_annotator_01.csv" in script
    assert "h200_qwen_full_sweep_audit_blinded_annotator_02.csv" in script
    assert "h200_causal_patch_qwen7b_audit_export_manifest.json" in script
    assert "h200_causal_patch_qwen7b_audit_blinded_annotator_01.csv" in script
    assert "h200_causal_patch_qwen7b_audit_blinded_annotator_02.csv" in script
    assert "rsync -az --checksum" in script
    assert "scripts/write_artifact_manifest.py" in script
    assert "scripts/compare_artifact_manifests.py" in script
    assert "git pull" not in script


def test_compare_artifact_manifests_reports_missing_paths() -> None:
    report = compare_manifests({"files": [], "missing_paths": ["results/run"]}, {"files": []})

    assert report["passed"] is False
    assert report["failures"] == ["missing_remote_path:results/run"]
