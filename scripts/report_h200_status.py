from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import utc_timestamp, write_json

EXPECTED_PATHS = [
    Path("results/h200_qwen_full_sweep"),
    Path("results/h200_causal_patch_qwen7b"),
    Path("results/h200_qwen32b_public_followup_primary"),
    Path("paper/generated/h200_qwen_full_sweep"),
    Path("paper/generated/h200_causal_patch_qwen7b"),
    Path("paper/generated/claim_assessment"),
    Path("paper/audit/h200_qwen_full_sweep_summary"),
    Path("paper/audit/h200_causal_patch_qwen7b_summary"),
]
PROCESS_PATTERNS = [
    "wait_and_run_h200_sweep",
    "wait_for_h200_gpu",
    "run_h200_sweep",
    "run_experiment.py",
]
WAIT_SAMPLE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) "
    r"memory\.used=(?P<memory>\d+)MiB utilization=(?P<utilization>\d+)%$"
)
WAIT_THRESHOLD_RE = re.compile(
    r"^Waiting for H200 GPU: memory\.used <= (?P<memory>\d+) MiB "
    r"and utilization <= (?P<utilization>\d+)%$"
)
DEFAULT_GATE_MEMORY_USED_MIB = 20_000
DEFAULT_GATE_UTILIZATION_PCT = 20
STALE_WAIT_SAMPLE_MINUTES = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Report H200 launcher, GPU, and artifact status.")
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--log-lines", type=int, default=40)
    args = parser.parse_args()

    status = h200_status(args.repo_dir, log_lines=args.log_lines)
    if args.output_json is not None:
        write_json(args.output_json, status)
    markdown = render_markdown(status)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


def h200_status(repo_dir: Path, *, log_lines: int = 40) -> dict[str, Any]:
    repo_dir = repo_dir.resolve()
    latest_log = _latest_launcher_log(repo_dir)
    gpu = _gpu_status()
    processes = _process_rows()
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "repo_dir": str(repo_dir),
        "git": {
            "commit": _command_text(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir),
            "status_short": _command_text(["git", "status", "--short"], cwd=repo_dir),
        },
        "processes": processes,
        "launcher_log": {
            "path": str(latest_log) if latest_log else None,
            "tail": _tail(latest_log, log_lines) if latest_log else "",
            "wait_history": _wait_history(latest_log) if latest_log else {"sample_count": 0},
        },
        "gpu": gpu,
        "expected_artifacts": _artifact_status(repo_dir),
        "experiment_running": any("run_experiment.py" in row["command"] for row in processes),
        "launcher_waiting": any("wait_for_h200_gpu" in row["command"] for row in processes),
        "gpu_gate_likely_blocked": _gpu_gate_likely_blocked(gpu),
        "gpu_gate_block_reasons": _gpu_gate_block_reasons(gpu),
        "hidden_gpu_context_likely": _hidden_gpu_context_likely(gpu),
    }


def render_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# H200 Status",
        "",
        f"Created: `{status['created_at_utc']}`",
        f"Repo: `{status['repo_dir']}`",
        f"Commit: `{status['git']['commit'] or 'unknown'}`",
        f"Experiment running: `{str(status['experiment_running']).lower()}`",
        f"Launcher waiting: `{str(status['launcher_waiting']).lower()}`",
        f"GPU gate likely blocked: `{str(status['gpu_gate_likely_blocked']).lower()}`",
        f"Hidden GPU context likely: `{str(status.get('hidden_gpu_context_likely', False)).lower()}`",
        "",
        "## GPU",
        "",
    ]
    gpu = status["gpu"]
    if gpu.get("available"):
        lines.append(
            f"- `{gpu['name']}` memory `{gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB`, "
            f"utilization `{gpu['utilization_pct']}%`"
        )
        reasons = status.get("gpu_gate_block_reasons") or []
        reason_text = ", ".join(reasons) if reasons else "none"
        lines.append(f"- gate block reasons: `{reason_text}`")
    else:
        lines.append(f"- unavailable: `{gpu.get('error')}`")
    if gpu.get("available"):
        lines.append("")
        lines.append("### Visible Compute Apps")
        lines.append("")
        compute_apps = gpu.get("compute_apps") or []
        if compute_apps:
            for app in compute_apps:
                lines.append(
                    f"- pid `{app['pid']}`, memory `{app['used_memory_mib']} MiB`: "
                    f"`{app['process_name']}`"
                )
        else:
            lines.append("- none reported by `nvidia-smi --query-compute-apps`")
        pmon = str(gpu.get("pmon") or "").strip()
        if pmon:
            lines.extend(["", "### Process Monitor Snapshot", "", "```text", pmon, "```"])
        device_holders = gpu.get("device_holders") or []
        lines.extend(["", "### Local NVIDIA Device Holders", ""])
        if device_holders:
            for holder in device_holders:
                lines.append(f"- pid `{holder['pid']}`: `{holder['command']}`")
        else:
            lines.append("- none found by scanning local `/proc/*/fd` for `/dev/nvidia*`")
        accounted_apps = gpu.get("accounted_apps") or []
        lines.extend(["", "### Accounted Apps", ""])
        if accounted_apps:
            for app in accounted_apps:
                lines.append(
                    f"- pid `{app['pid']}`, gpu `{app['gpu_name']}`, time `{app['time']}`"
                )
        else:
            lines.append("- none reported by `nvidia-smi --query-accounted-apps`")
        pid_query = str(gpu.get("pid_query") or "").strip()
        if pid_query:
            lines.extend(["", "### NVIDIA PIDS Query", "", "```text", pid_query, "```"])
        if status.get("hidden_gpu_context_likely"):
            lines.extend(
                [
                    "",
                    "### Blocker Diagnosis",
                    "",
                    "- GPU memory or utilization is high, but `nvidia-smi` reports no visible compute app. "
                    "The blocker may be outside this namespace/cgroup, a stale driver context, "
                    "or a notebook-level reservation.",
                    "- Non-destructive next step: keep the launcher running, release or restart the "
                    "notebook allocation from the UI if this is your session, then rerun this status "
                    "report. Avoid `nvidia-smi --gpu-reset` or killing unknown processes on shared "
                    "infrastructure unless an administrator explicitly authorizes it.",
                ]
            )
    lines.extend(["", "## Processes", ""])
    if status["processes"]:
        for row in status["processes"]:
            lines.append(f"- pid `{row['pid']}`, elapsed `{row['elapsed']}`: `{row['command']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Expected Artifacts", ""])
    for row in status["expected_artifacts"]:
        state = "present" if row["exists"] else "missing"
        lines.append(f"- `{row['path']}`: {state}")
    wait_history = status["launcher_log"].get("wait_history") or {}
    if wait_history.get("sample_count"):
        first = wait_history["first"]
        latest = wait_history["latest"]
        minimum = wait_history["min_memory"]
        threshold = wait_history.get("gate_threshold") or {}
        plateau = wait_history.get("latest_memory_plateau") or {}
        lines.extend(
            [
                "",
                "## Wait History",
                "",
                f"- samples: `{wait_history['sample_count']}`",
                f"- observed wait duration: `{float(wait_history.get('observed_wait_minutes') or 0.0):.1f} minutes`",
                (
                    f"- gate threshold: memory `<= {threshold.get('memory_used_mib', 'unknown')} MiB`, "
                    f"utilization `<= {threshold.get('utilization_pct', 'unknown')}%`"
                ),
                (
                    f"- first sample: `{first['timestamp_utc']}` "
                    f"`{first['memory_used_mib']} MiB`, `{first['utilization_pct']}%`"
                ),
                (
                    f"- latest sample: `{latest['timestamp_utc']}` "
                    f"`{latest['memory_used_mib']} MiB`, `{latest['utilization_pct']}%`"
                ),
                (
                    f"- minimum memory observed: `{minimum['memory_used_mib']} MiB` "
                    f"at `{minimum['timestamp_utc']}`"
                ),
                (
                    f"- latest memory plateau: `{plateau.get('memory_used_mib', 'unknown')} MiB` "
                    f"for `{plateau.get('sample_count', 'unknown')}` samples "
                    f"since `{plateau.get('first_seen_utc', 'unknown')}` "
                    f"({float(plateau.get('duration_minutes') or 0.0):.1f} minutes)"
                ),
                f"- memory drop from first to latest: `{wait_history['memory_drop_mib']} MiB`",
                f"- latest sample passes gate: `{str(wait_history['latest_gate_passed']).lower()}`",
                f"- prolonged gate block: `{str(wait_history.get('prolonged_gate_block', False)).lower()}`",
                f"- latest sample age: `{float(wait_history.get('latest_sample_age_minutes') or 0.0):.1f} minutes`",
                f"- launcher log stale: `{str(wait_history.get('launcher_log_stale', False)).lower()}`",
            ]
        )
    lines.extend(["", "## Latest Launcher Log", ""])
    if status["launcher_log"]["path"]:
        lines.append(f"`{status['launcher_log']['path']}`")
        lines.append("")
        lines.append("```text")
        lines.append(status["launcher_log"]["tail"].rstrip())
        lines.append("```")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _gpu_status() -> dict[str, Any]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=None,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or result.stdout.strip()}
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parsed = _parse_gpu_query_line(line)
    if parsed is None:
        return {"available": False, "error": f"could not parse nvidia-smi output: {line}"}
    pmon = _run(["nvidia-smi", "pmon", "-c", "1"], cwd=None)
    parsed["pmon"] = pmon.stdout.strip() if pmon.returncode == 0 else pmon.stderr.strip()
    parsed["compute_apps"] = _compute_apps()
    parsed["accounted_apps"] = _accounted_apps()
    parsed["pid_query"] = _nvidia_pid_query()
    parsed["device_holders"] = _nvidia_device_holders()
    return parsed


def _parse_gpu_query_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        return {
            "available": True,
            "name": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_free_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
            "utilization_pct": int(parts[4]),
        }
    except ValueError:
        return None


def _gpu_gate_likely_blocked(gpu: dict[str, Any]) -> bool:
    return bool(_gpu_gate_block_reasons(gpu))


def _gpu_gate_block_reasons(gpu: dict[str, Any]) -> list[str]:
    if not gpu.get("available"):
        return []
    reasons = []
    if int(gpu.get("memory_used_mib") or 0) > DEFAULT_GATE_MEMORY_USED_MIB:
        reasons.append("memory_used")
    if int(gpu.get("utilization_pct") or 0) > DEFAULT_GATE_UTILIZATION_PCT:
        reasons.append("utilization")
    return reasons


def _hidden_gpu_context_likely(gpu: dict[str, Any]) -> bool:
    if not _gpu_gate_likely_blocked(gpu):
        return False
    if gpu.get("compute_apps"):
        return False
    if gpu.get("device_holders"):
        return False
    return "No running processes found" in str(gpu.get("pmon") or "") or _pmon_has_no_process_rows(
        str(gpu.get("pmon") or "")
    )


def _pmon_has_no_process_rows(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) > 1 and parts[1] != "-":
            return False
    return True


def _compute_apps() -> list[dict[str, Any]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=None,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parsed = _parse_compute_app_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _accounted_apps() -> list[dict[str, str]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-accounted-apps=pid,gpu_name,time",
            "--format=csv,noheader,nounits",
        ],
        cwd=None,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parsed = _parse_accounted_app_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _parse_accounted_app_line(line: str) -> dict[str, str] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3 or not parts[0]:
        return None
    return {"pid": parts[0], "gpu_name": parts[1], "time": parts[2]}


def _nvidia_pid_query() -> str:
    result = _run(["nvidia-smi", "-q", "-d", "PIDS"], cwd=None)
    if result.returncode != 0:
        return result.stderr.strip()
    return _trim_diagnostic_output(result.stdout, max_lines=80)


def _trim_diagnostic_output(text: str, *, max_lines: int) -> str:
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join([*lines[:max_lines], "... truncated ..."])


def _parse_compute_app_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3 or not parts[0]:
        return None
    try:
        used_memory_mib = int(parts[2])
    except ValueError:
        return None
    return {"pid": parts[0], "process_name": parts[1], "used_memory_mib": used_memory_mib}


def _nvidia_device_holders(
    proc_root: Path = Path("/proc"),
    *,
    limit: int = 25,
) -> list[dict[str, str]]:
    holders = []
    for process_dir in sorted(proc_root.glob("[0-9]*"), key=lambda path: int(path.name)):
        fd_dir = process_dir / "fd"
        if not fd_dir.exists():
            continue
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        if not any(_fd_points_to_nvidia_device(fd) for fd in fds):
            continue
        holders.append({"pid": process_dir.name, "command": _proc_command(process_dir)})
        if len(holders) >= limit:
            break
    return holders


def _fd_points_to_nvidia_device(fd: Path) -> bool:
    try:
        target = os.readlink(fd)
    except OSError:
        return False
    return target.startswith("/dev/nvidia")


def _proc_command(process_dir: Path) -> str:
    try:
        raw = (process_dir / "cmdline").read_bytes()
    except OSError:
        raw = b""
    command = " ".join(
        part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
    )
    if command:
        return command
    try:
        return (process_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "<unknown>"


def _process_rows() -> list[dict[str, str]]:
    result = _run(["ps", "-eo", "pid,ppid,stat,etime,cmd"], cwd=None)
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines()[1:]:
        if not any(pattern in line for pattern in PROCESS_PATTERNS):
            continue
        if _is_status_probe_process(line):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": parts[0],
                "ppid": parts[1],
                "stat": parts[2],
                "elapsed": parts[3],
                "command": parts[4],
            }
        )
    return rows


def _is_status_probe_process(line: str) -> bool:
    probe_markers = [
        "report_h200_status.py",
        "ps -eo",
        "grep -E",
        "nvidia-smi --query",
    ]
    return any(marker in line for marker in probe_markers)


def _artifact_status(repo_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for relative_path in EXPECTED_PATHS:
        path = repo_dir / relative_path
        rows.append({"path": str(relative_path), "exists": path.exists()})
    return rows


def _latest_launcher_log(repo_dir: Path) -> Path | None:
    logs = sorted((repo_dir / "logs" / "h200").glob("wait_and_run_*.log"), reverse=True)
    return logs[0] if logs else None


def _wait_history(path: Path, *, current_utc: str | None = None) -> dict[str, Any]:
    samples = []
    threshold = {
        "memory_used_mib": DEFAULT_GATE_MEMORY_USED_MIB,
        "utilization_pct": DEFAULT_GATE_UTILIZATION_PCT,
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed_threshold = _parse_wait_threshold(line)
        if parsed_threshold is not None:
            threshold = parsed_threshold
        sample = _parse_wait_sample(line)
        if sample is not None:
            samples.append(sample)
    if not samples:
        return {"sample_count": 0}
    first = samples[0]
    latest = samples[-1]
    min_memory = min(samples, key=lambda item: item["memory_used_mib"])
    max_memory = max(samples, key=lambda item: item["memory_used_mib"])
    latest_plateau = _latest_memory_plateau(samples)
    observed_wait_minutes = _minutes_between(first["timestamp_utc"], latest["timestamp_utc"])
    latest_sample_age_minutes = _latest_sample_age_minutes(
        latest["timestamp_utc"], current_utc=current_utc
    )
    latest_gate_passed = latest["memory_used_mib"] <= threshold["memory_used_mib"] and (
        latest["utilization_pct"] <= threshold["utilization_pct"]
    )
    return {
        "sample_count": len(samples),
        "first": first,
        "latest": latest,
        "min_memory": min_memory,
        "max_memory": max_memory,
        "latest_memory_plateau": latest_plateau,
        "observed_wait_minutes": observed_wait_minutes,
        "min_utilization_pct": min(item["utilization_pct"] for item in samples),
        "max_utilization_pct": max(item["utilization_pct"] for item in samples),
        "memory_drop_mib": first["memory_used_mib"] - latest["memory_used_mib"],
        "gate_threshold": threshold,
        "latest_gate_passed": latest_gate_passed,
        "prolonged_gate_block": not latest_gate_passed and observed_wait_minutes >= 60.0,
        "latest_sample_age_minutes": latest_sample_age_minutes,
        "launcher_log_stale": latest_sample_age_minutes >= STALE_WAIT_SAMPLE_MINUTES,
    }


def _latest_memory_plateau(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latest = samples[-1]
    latest_memory = latest["memory_used_mib"]
    plateau = []
    for sample in reversed(samples):
        if sample["memory_used_mib"] != latest_memory:
            break
        plateau.append(sample)
    first_plateau_sample = plateau[-1]
    return {
        "memory_used_mib": latest_memory,
        "sample_count": len(plateau),
        "first_seen_utc": first_plateau_sample["timestamp_utc"],
        "latest_seen_utc": latest["timestamp_utc"],
        "duration_minutes": _minutes_between(
            first_plateau_sample["timestamp_utc"], latest["timestamp_utc"]
        ),
    }


def _minutes_between(start_utc: str, end_utc: str) -> float:
    start = _parse_utc_timestamp(start_utc)
    end = _parse_utc_timestamp(end_utc)
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 60.0)


def _latest_sample_age_minutes(latest_utc: str, *, current_utc: str | None) -> float:
    latest = _parse_utc_timestamp(latest_utc)
    if latest is None:
        return 0.0
    current = _parse_utc_timestamp(current_utc) if current_utc is not None else None
    if current is None:
        current = datetime.now(UTC)
    return max(0.0, (current - latest).total_seconds() / 60.0)


def _parse_utc_timestamp(timestamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_wait_threshold(line: str) -> dict[str, int] | None:
    match = WAIT_THRESHOLD_RE.match(line.strip())
    if match is None:
        return None
    return {
        "memory_used_mib": int(match.group("memory")),
        "utilization_pct": int(match.group("utilization")),
    }


def _parse_wait_sample(line: str) -> dict[str, Any] | None:
    match = WAIT_SAMPLE_RE.match(line.strip())
    if match is None:
        return None
    return {
        "timestamp_utc": match.group("timestamp"),
        "memory_used_mib": int(match.group("memory")),
        "utilization_pct": int(match.group("utilization")),
    }


def _tail(path: Path, line_count: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _command_text(command: list[str], *, cwd: Path) -> str:
    result = _run(command, cwd=cwd)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _run(
    command: list[str],
    *,
    cwd: Path | None,
    timeout_seconds: float = 20,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = stderr.strip()
        if message:
            message = f"{message}\n"
        message += f"timed out after {timeout_seconds:g}s: {' '.join(command)}"
        return subprocess.CompletedProcess(command, 124, stdout, message)


if __name__ == "__main__":
    main()
