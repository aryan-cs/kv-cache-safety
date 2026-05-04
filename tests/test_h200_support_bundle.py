import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from package_h200_support_bundle import package_support_bundle, support_bundle_inputs


def test_support_bundle_packages_only_infrastructure_diagnostics(tmp_path: Path) -> None:
    status_json = tmp_path / "h200_status_latest.json"
    status_md = tmp_path / "h200_status_latest.md"
    admin_md = tmp_path / "h200_admin_report.md"
    launcher_log = tmp_path / "wait_and_run.log"
    output = tmp_path / "support.tar.gz"
    status_json.write_text(
        json.dumps(
            {
                "created_at_utc": "20260504T000000Z",
                "repo_dir": "/home/aryang9/sandbox/llm-safety",
                "git": {"commit": "abc123"},
                "launcher_waiting": True,
                "gpu_gate_block_reasons": ["utilization"],
                "hidden_gpu_context_likely": True,
                "launcher_log": {"path": str(launcher_log)},
            }
        ),
        encoding="utf-8",
    )
    status_md.write_text("# status\n", encoding="utf-8")
    admin_md.write_text("# admin\n", encoding="utf-8")
    launcher_log.write_text("Waiting for H200 GPU\n", encoding="utf-8")

    package_support_bundle(
        status_json=status_json,
        status_md=status_md,
        admin_md=admin_md,
        output=output,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = sorted(archive.getnames())
        manifest = json.loads(archive.extractfile("manifest.json").read().decode("utf-8"))

    assert names == [
        "h200_admin_report.md",
        "h200_status_latest.json",
        "h200_status_latest.md",
        "launcher_log_tail_source.log",
        "manifest.json",
    ]
    assert manifest["contains_model_generations"] is False
    assert manifest["contains_paper_evidence"] is False
    assert manifest["hidden_gpu_context_likely"] is True


def test_support_bundle_inputs_skip_missing_optional_files(tmp_path: Path) -> None:
    status_json = tmp_path / "h200_status_latest.json"
    status_json.write_text("{}", encoding="utf-8")

    inputs = support_bundle_inputs(
        {"launcher_log": {"path": str(tmp_path / "missing.log")}},
        status_json=status_json,
        status_md=tmp_path / "missing.md",
        admin_md=tmp_path / "missing_admin.md",
    )

    assert inputs == [("h200_status_latest.json", status_json)]


def test_support_bundle_requires_status_json(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        package_support_bundle(
            status_json=tmp_path / "missing.json",
            status_md=tmp_path / "status.md",
            admin_md=tmp_path / "admin.md",
            output=tmp_path / "bundle.tar.gz",
        )

    assert "Run scripts/report_h200_status.py with --output-json first" in str(excinfo.value)
