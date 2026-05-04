import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from report_h200_status import (
    _artifact_status,
    _gpu_gate_likely_blocked,
    _is_status_probe_process,
    _latest_memory_plateau,
    _nvidia_device_holders,
    _parse_accounted_app_line,
    _parse_compute_app_line,
    _parse_gpu_query_line,
    _parse_wait_sample,
    _parse_wait_threshold,
    _run,
    _trim_diagnostic_output,
    _wait_history,
    render_markdown,
)


def test_parse_gpu_query_line() -> None:
    parsed = _parse_gpu_query_line("NVIDIA H200 NVL, 142461, 707, 143771, 100")

    assert parsed == {
        "available": True,
        "name": "NVIDIA H200 NVL",
        "memory_used_mib": 142461,
        "memory_free_mib": 707,
        "memory_total_mib": 143771,
        "utilization_pct": 100,
    }
    assert _gpu_gate_likely_blocked(parsed)


def test_parse_gpu_query_line_rejects_malformed_output() -> None:
    assert _parse_gpu_query_line("not enough fields") is None


def test_parse_compute_app_line() -> None:
    parsed = _parse_compute_app_line("1234, python, 4096")

    assert parsed == {"pid": "1234", "process_name": "python", "used_memory_mib": 4096}
    assert _parse_compute_app_line("not enough fields") is None
    assert _parse_compute_app_line("1234, python, N/A") is None


def test_parse_wait_sample() -> None:
    parsed = _parse_wait_sample("2026-05-04T03:59:11Z memory.used=82139MiB utilization=95%")

    assert parsed == {
        "timestamp_utc": "2026-05-04T03:59:11Z",
        "memory_used_mib": 82139,
        "utilization_pct": 95,
    }
    assert _parse_wait_sample("not a wait sample") is None


def test_parse_wait_threshold() -> None:
    parsed = _parse_wait_threshold(
        "Waiting for H200 GPU: memory.used <= 20000 MiB and utilization <= 20%"
    )

    assert parsed == {"memory_used_mib": 20000, "utilization_pct": 20}
    assert _parse_wait_threshold("not a threshold") is None


def test_wait_history_summarizes_launcher_memory_trend(tmp_path: Path) -> None:
    log = tmp_path / "wait.log"
    log.write_text(
        "\n".join(
            [
                "Waiting for H200 GPU: memory.used <= 20000 MiB and utilization <= 20%",
                "2026-05-04T02:04:08Z memory.used=142461MiB utilization=100%",
                "2026-05-04T03:59:11Z memory.used=82139MiB utilization=95%",
                "2026-05-04T04:04:11Z memory.used=19000MiB utilization=15%",
            ]
        ),
        encoding="utf-8",
    )

    history = _wait_history(log)

    assert history["sample_count"] == 3
    assert history["first"]["memory_used_mib"] == 142461
    assert history["latest"]["memory_used_mib"] == 19000
    assert history["min_memory"]["memory_used_mib"] == 19000
    assert history["memory_drop_mib"] == 123461
    assert history["gate_threshold"] == {"memory_used_mib": 20000, "utilization_pct": 20}
    assert history["observed_wait_minutes"] == pytest.approx(120.05)
    assert history["latest_memory_plateau"] == {
        "memory_used_mib": 19000,
        "sample_count": 1,
        "first_seen_utc": "2026-05-04T04:04:11Z",
        "latest_seen_utc": "2026-05-04T04:04:11Z",
        "duration_minutes": 0.0,
    }
    assert history["latest_gate_passed"] is True
    assert history["prolonged_gate_block"] is False


def test_wait_history_marks_prolonged_gate_block_only_while_blocked(tmp_path: Path) -> None:
    log = tmp_path / "wait.log"
    log.write_text(
        "\n".join(
            [
                "Waiting for H200 GPU: memory.used <= 20000 MiB and utilization <= 20%",
                "2026-05-04T02:04:08Z memory.used=142461MiB utilization=100%",
                "2026-05-04T03:59:11Z memory.used=82139MiB utilization=95%",
            ]
        ),
        encoding="utf-8",
    )

    history = _wait_history(log)

    assert history["latest_gate_passed"] is False
    assert history["observed_wait_minutes"] == pytest.approx(115.05)
    assert history["prolonged_gate_block"] is True


def test_wait_history_prolonged_gate_block_starts_at_sixty_minutes(
    tmp_path: Path,
) -> None:
    log = tmp_path / "wait.log"
    log.write_text(
        "\n".join(
            [
                "Waiting for H200 GPU: memory.used <= 20000 MiB and utilization <= 20%",
                "2026-05-04T02:04:08Z memory.used=82139MiB utilization=100%",
                "2026-05-04T03:04:07Z memory.used=82139MiB utilization=100%",
            ]
        ),
        encoding="utf-8",
    )

    history = _wait_history(log)

    assert history["observed_wait_minutes"] == pytest.approx(59.9833333333)
    assert history["prolonged_gate_block"] is False

    log.write_text(
        "\n".join(
            [
                "Waiting for H200 GPU: memory.used <= 20000 MiB and utilization <= 20%",
                "2026-05-04T02:04:08Z memory.used=82139MiB utilization=100%",
                "2026-05-04T03:04:08Z memory.used=82139MiB utilization=100%",
            ]
        ),
        encoding="utf-8",
    )

    history = _wait_history(log)

    assert history["observed_wait_minutes"] == 60.0
    assert history["prolonged_gate_block"] is True


def test_latest_memory_plateau_tracks_repeated_latest_memory() -> None:
    plateau = _latest_memory_plateau(
        [
            {
                "timestamp_utc": "2026-05-04T03:54:11Z",
                "memory_used_mib": 142461,
                "utilization_pct": 100,
            },
            {
                "timestamp_utc": "2026-05-04T03:59:11Z",
                "memory_used_mib": 82139,
                "utilization_pct": 95,
            },
            {
                "timestamp_utc": "2026-05-04T04:04:11Z",
                "memory_used_mib": 82139,
                "utilization_pct": 100,
            },
        ]
    )

    assert plateau == {
        "memory_used_mib": 82139,
        "sample_count": 2,
        "first_seen_utc": "2026-05-04T03:59:11Z",
        "latest_seen_utc": "2026-05-04T04:04:11Z",
        "duration_minutes": 5.0,
    }


def test_wait_history_uses_custom_gate_threshold(tmp_path: Path) -> None:
    log = tmp_path / "wait.log"
    log.write_text(
        "\n".join(
            [
                "Waiting for H200 GPU: memory.used <= 90000 MiB and utilization <= 99%",
                "2026-05-04T03:59:11Z memory.used=82139MiB utilization=95%",
            ]
        ),
        encoding="utf-8",
    )

    history = _wait_history(log)

    assert history["gate_threshold"] == {"memory_used_mib": 90000, "utilization_pct": 99}
    assert history["latest_gate_passed"] is True


def test_parse_accounted_app_line() -> None:
    parsed = _parse_accounted_app_line("1234, NVIDIA H200 NVL, 00:05:00")

    assert parsed == {"pid": "1234", "gpu_name": "NVIDIA H200 NVL", "time": "00:05:00"}
    assert _parse_accounted_app_line("not enough fields") is None


def test_trim_diagnostic_output_bounds_long_text() -> None:
    text = "\n".join(f"line {index}" for index in range(5))

    assert _trim_diagnostic_output(text, max_lines=3) == "line 0\nline 1\nline 2\n... truncated ..."


def test_status_probe_process_filter_skips_monitoring_shells() -> None:
    assert _is_status_probe_process("bash -c ps -eo pid,ppid,stat,etime,cmd | grep -E wait")
    assert _is_status_probe_process("python scripts/report_h200_status.py")
    assert not _is_status_probe_process("bash scripts/wait_for_h200_gpu.sh")


def test_nvidia_device_holders_scan_proc_fd_links(tmp_path: Path) -> None:
    process_dir = tmp_path / "123"
    fd_dir = process_dir / "fd"
    fd_dir.mkdir(parents=True)
    (process_dir / "cmdline").write_bytes(b"python\0train.py\0")
    (fd_dir / "4").symlink_to("/dev/nvidia0")

    assert _nvidia_device_holders(tmp_path) == [
        {"pid": "123", "command": "python train.py"}
    ]


def test_run_reports_missing_executable() -> None:
    result = _run(["definitely_missing_h200_status_binary"], cwd=None)

    assert result.returncode == 127
    assert "definitely_missing_h200_status_binary" in result.stderr


def test_run_reports_command_timeout() -> None:
    result = _run(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        cwd=None,
        timeout_seconds=0.01,
    )

    assert result.returncode == 124
    assert "timed out after" in result.stderr


def test_artifact_status_marks_expected_h200_dirs(tmp_path: Path) -> None:
    (tmp_path / "results" / "h200_qwen_full_sweep").mkdir(parents=True)

    rows = _artifact_status(tmp_path)

    lookup = {row["path"]: row["exists"] for row in rows}
    assert lookup["results/h200_qwen_full_sweep"] is True
    assert lookup["results/h200_causal_patch_qwen7b"] is False


def test_render_markdown_summarizes_blocked_launcher() -> None:
    text = render_markdown(
        {
            "created_at_utc": "20260504T000000Z",
            "repo_dir": "/home/aryang9/sandbox/llm-safety",
            "git": {"commit": "abc123"},
            "experiment_running": False,
            "launcher_waiting": True,
            "gpu_gate_likely_blocked": True,
            "hidden_gpu_context_likely": True,
            "gpu": {
                "available": True,
                "name": "NVIDIA H200 NVL",
                "memory_used_mib": 142461,
                "memory_total_mib": 143771,
                "utilization_pct": 100,
                "compute_apps": [],
                "accounted_apps": [],
                "pid_query": "Processes                             : None",
                "device_holders": [],
                "pmon": "# gpu pid type sm mem\n0 - - - -",
            },
            "processes": [
                {
                    "pid": "10",
                    "elapsed": "01:00",
                    "command": "bash scripts/wait_for_h200_gpu.sh",
                }
            ],
            "expected_artifacts": [{"path": "results/h200_qwen_full_sweep", "exists": False}],
            "launcher_log": {
                "path": "logs/h200/wait.log",
                "tail": "Waiting for H200 GPU",
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
                },
            },
        }
    )

    assert "GPU gate likely blocked: `true`" in text
    assert "none reported by `nvidia-smi --query-compute-apps`" in text
    assert "Process Monitor Snapshot" in text
    assert "none found by scanning local `/proc/*/fd`" in text
    assert "none reported by `nvidia-smi --query-accounted-apps`" in text
    assert "NVIDIA PIDS Query" in text
    assert "Wait History" in text
    assert "observed wait duration: `115.0 minutes`" in text
    assert "gate threshold: memory `<= 20000 MiB`, utilization `<= 20%`" in text
    assert "latest memory plateau: `82139 MiB` for `2` samples" in text
    assert "memory drop from first to latest: `60322 MiB`" in text
    assert "prolonged gate block: `true`" in text
    assert "release or restart the notebook allocation" in text
    assert "`results/h200_qwen_full_sweep`: missing" in text
