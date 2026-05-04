import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from write_h200_admin_report import admin_report, main


def test_admin_report_summarizes_hidden_gpu_context_without_results() -> None:
    report = admin_report(
        {
            "created_at_utc": "20260504T000000Z",
            "repo_dir": "/home/aryang9/sandbox/llm-safety",
            "git": {"commit": "abc123"},
            "experiment_running": False,
            "launcher_waiting": True,
            "gpu_gate_likely_blocked": True,
            "gpu_gate_block_reasons": ["utilization"],
            "hidden_gpu_context_likely": True,
            "gpu": {
                "available": True,
                "name": "NVIDIA H200 NVL",
                "memory_used_mib": 142461,
                "memory_total_mib": 143771,
                "utilization_pct": 100,
                "compute_apps": [],
                "accounted_apps": [],
                "device_holders": [],
                "pmon": "# gpu pid type sm mem\n0 - - - -",
                "pid_query": "Processes                             : None",
            },
            "processes": [
                {
                    "pid": "10",
                    "elapsed": "01:00",
                    "command": "bash scripts/wait_for_h200_gpu.sh",
                }
            ],
            "launcher_log": {
                "wait_history": {
                    "sample_count": 2,
                    "first": {
                        "timestamp_utc": "2026-05-04T02:04:08Z",
                        "memory_used_mib": 142461,
                        "utilization_pct": 100,
                    },
                    "latest": {
                        "timestamp_utc": "2026-05-04T03:59:11Z",
                        "memory_used_mib": 82139,
                        "utilization_pct": 95,
                    },
                    "min_memory": {
                        "timestamp_utc": "2026-05-04T03:59:11Z",
                        "memory_used_mib": 82139,
                        "utilization_pct": 95,
                    },
                    "observed_wait_minutes": 115.0,
                    "memory_drop_mib": 60322,
                    "gate_threshold": {"memory_used_mib": 20000, "utilization_pct": 20},
                    "latest_memory_plateau": {
                        "memory_used_mib": 82139,
                        "sample_count": 2,
                        "first_seen_utc": "2026-05-04T03:59:11Z",
                        "latest_seen_utc": "2026-05-04T04:04:11Z",
                        "duration_minutes": 5.0,
                    },
                    "latest_gate_passed": False,
                    "prolonged_gate_block": True,
                }
            },
        }
    )

    assert "infrastructure diagnostics only" in report
    assert "Hidden GPU context likely: `true`" in report
    assert "GPU gate block reasons: `utilization`" in report
    assert "Visible compute apps: `0`" in report
    assert "Processes                             : None" in report
    assert "Wait History" in report
    assert "Observed wait duration: `115.0 minutes`" in report
    assert "Gate threshold: memory `<= 20000 MiB`, utilization `<= 20%`" in report
    assert "Latest memory plateau: `82139 MiB` for `2` samples" in report
    assert "Memory drop from first to latest: `60322 MiB`" in report
    assert "Prolonged gate block: `true`" in report
    assert "release or restart the notebook allocation" in report
    assert "nvidia-smi --gpu-reset" in report
    assert "model generations" in report


def test_admin_report_cli_reports_missing_status_json() -> None:
    original_argv = sys.argv
    sys.argv = [
        "write_h200_admin_report.py",
        "--status-json",
        "logs/h200/does_not_exist.json",
    ]
    try:
        with pytest.raises(SystemExit) as excinfo:
            main()
    finally:
        sys.argv = original_argv

    assert "Run scripts/report_h200_status.py with --output-json first" in str(excinfo.value)
