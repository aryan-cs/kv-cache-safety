from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a concise support report for hidden H200 GPU-context blockers."
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=Path("logs/h200/h200_status_latest.json"),
        help="JSON produced by scripts/report_h200_status.py.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("logs/h200/h200_admin_report.md"),
    )
    args = parser.parse_args()

    if not args.status_json.exists():
        raise SystemExit(
            f"Missing status JSON: {args.status_json}. "
            "Run scripts/report_h200_status.py with --output-json first."
        )
    status = json.loads(args.status_json.read_text(encoding="utf-8"))
    report = admin_report(status)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_md}")


def admin_report(status: dict[str, Any]) -> str:
    gpu = status.get("gpu", {})
    lines = [
        "# H200 Hidden GPU Context Support Report",
        "",
        "This report contains infrastructure diagnostics only. It is not an experiment result "
        "and contains no model generations, labels, or paper evidence.",
        "",
        "## Summary",
        "",
        f"- Created: `{status.get('created_at_utc', 'unknown')}`",
        f"- Repo: `{status.get('repo_dir', 'unknown')}`",
        f"- Git commit: `{status.get('git', {}).get('commit', 'unknown')}`",
        f"- Experiment running: `{_bool(status.get('experiment_running'))}`",
        f"- Launcher waiting: `{_bool(status.get('launcher_waiting'))}`",
        f"- GPU gate blocked: `{_bool(status.get('gpu_gate_likely_blocked'))}`",
        f"- GPU gate block reasons: `{_reasons(status.get('gpu_gate_block_reasons'))}`",
        f"- Hidden GPU context likely: `{_bool(status.get('hidden_gpu_context_likely'))}`",
        "",
        "## GPU Snapshot",
        "",
    ]
    if gpu.get("available"):
        lines.extend(
            [
                f"- GPU: `{gpu.get('name')}`",
                (
                    f"- Memory used: `{gpu.get('memory_used_mib')}/"
                    f"{gpu.get('memory_total_mib')} MiB`"
                ),
                f"- Utilization: `{gpu.get('utilization_pct')}%`",
            ]
        )
    else:
        lines.append(f"- GPU unavailable to status script: `{gpu.get('error', 'unknown')}`")

    lines.extend(
        [
            "",
            "## Evidence Of Hidden Or Stale Context",
            "",
            f"- Visible compute apps: `{_count(gpu.get('compute_apps'))}`",
            f"- Accounted apps: `{_count(gpu.get('accounted_apps'))}`",
            f"- Local `/proc/*/fd` NVIDIA holders: `{_count(gpu.get('device_holders'))}`",
            "",
        ]
    )
    pmon = str(gpu.get("pmon") or "").strip()
    if pmon:
        lines.extend(["### `nvidia-smi pmon -c 1`", "", "```text", pmon, "```", ""])
    pid_query = str(gpu.get("pid_query") or "").strip()
    if pid_query:
        lines.extend(["### `nvidia-smi -q -d PIDS`", "", "```text", pid_query, "```", ""])

    wait_history = status.get("launcher_log", {}).get("wait_history") or {}
    if wait_history.get("sample_count"):
        first = wait_history["first"]
        latest = wait_history["latest"]
        minimum = wait_history["min_memory"]
        threshold = wait_history.get("gate_threshold") or {}
        plateau = wait_history.get("latest_memory_plateau") or {}
        block_window = wait_history.get("latest_gate_block_window") or {}
        lines.extend(
            [
                "## Wait History",
                "",
                f"- Samples: `{wait_history['sample_count']}`",
                f"- Observed wait duration: `{float(wait_history.get('observed_wait_minutes') or 0.0):.1f} minutes`",
                (
                    f"- Gate threshold: memory `<= {threshold.get('memory_used_mib', 'unknown')} MiB`, "
                    f"utilization `<= {threshold.get('utilization_pct', 'unknown')}%`"
                ),
                (
                    f"- First sample: `{first['timestamp_utc']}` "
                    f"`{first['memory_used_mib']} MiB`, `{first['utilization_pct']}%`"
                ),
                (
                    f"- Latest sample: `{latest['timestamp_utc']}` "
                    f"`{latest['memory_used_mib']} MiB`, `{latest['utilization_pct']}%`"
                ),
                (
                    f"- Minimum memory observed: `{minimum['memory_used_mib']} MiB` "
                    f"at `{minimum['timestamp_utc']}`"
                ),
                (
                    f"- Latest memory plateau: `{plateau.get('memory_used_mib', 'unknown')} MiB` "
                    f"for `{plateau.get('sample_count', 'unknown')}` samples "
                    f"since `{plateau.get('first_seen_utc', 'unknown')}` "
                    f"({float(plateau.get('duration_minutes') or 0.0):.1f} minutes)"
                ),
                f"- Memory drop from first to latest: `{wait_history['memory_drop_mib']} MiB`",
                f"- Latest sample passes gate: `{_bool(wait_history['latest_gate_passed'])}`",
                f"- Prolonged gate block: `{_bool(wait_history.get('prolonged_gate_block'))}`",
                (
                    f"- Latest gate block window: `{block_window.get('reason', 'unknown')}` "
                    f"for `{block_window.get('sample_count', 0)}` samples "
                    f"since `{block_window.get('first_seen_utc', 'n/a')}` "
                    f"({float(block_window.get('duration_minutes') or 0.0):.1f} minutes)"
                ),
                f"- Latest sample age: `{float(wait_history.get('latest_sample_age_minutes') or 0.0):.1f} minutes`",
                f"- Launcher log stale: `{_bool(wait_history.get('launcher_log_stale'))}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Waiting Processes",
            "",
        ]
    )
    processes = status.get("processes") or []
    if processes:
        for row in processes:
            lines.append(
                f"- pid `{row.get('pid')}`, elapsed `{row.get('elapsed')}`: "
                f"`{row.get('command')}`"
            )
    else:
        lines.append("- none")

    lines.extend(_copy_paste_support_request(status, gpu))

    lines.extend(_requested_non_destructive_action(status, gpu))
    return "\n".join(lines)


def _copy_paste_support_request(status: dict[str, Any], gpu: dict[str, Any]) -> list[str]:
    memory_used = gpu.get("memory_used_mib", "unknown")
    memory_total = gpu.get("memory_total_mib", "unknown")
    utilization = gpu.get("utilization_pct", "unknown")
    threshold = (
        status.get("launcher_log", {})
        .get("wait_history", {})
        .get("gate_threshold", {})
    )
    memory_threshold = threshold.get("memory_used_mib", 20000)
    utilization_threshold = threshold.get("utilization_pct", 20)
    support_text = _support_request_body(
        status,
        gpu,
        memory_used=memory_used,
        memory_total=memory_total,
        utilization=utilization,
        memory_threshold=memory_threshold,
        utilization_threshold=utilization_threshold,
    )
    return [
        "",
        "## Copy/Paste Support Request",
        "",
        "```text",
        *support_text,
        "```",
    ]


def _support_request_body(
    status: dict[str, Any],
    gpu: dict[str, Any],
    *,
    memory_used: Any,
    memory_total: Any,
    utilization: Any,
    memory_threshold: Any,
    utilization_threshold: Any,
) -> list[str]:
    common = [
        "",
        f"Repo: {status.get('repo_dir', 'unknown')}",
        f"Git commit: {status.get('git', {}).get('commit', 'unknown')}",
        f"Launcher waiting: {_bool(status.get('launcher_waiting'))}",
        f"Experiment running: {_bool(status.get('experiment_running'))}",
        "Support bundle: logs/h200/h200_support_bundle_latest.tar.gz",
    ]
    if not gpu.get("available"):
        return [
            "Hello, my H200 notebook allocation cannot reliably query the GPU "
            "from inside the notebook environment.",
            *common,
            f"GPU status: unavailable ({gpu.get('error', 'unknown')})",
            "Please check whether the H200 allocation is attached and visible to "
            "the notebook before any cleanup or restart action.",
            "We have not run nvidia-smi --gpu-reset or killed unknown processes.",
        ]

    visible_holder_count = _visible_holder_count(gpu)
    stale_note = _launcher_stale_note(status)
    if _hidden_context_case(status, gpu):
        body = [
            "Hello, my H200 notebook allocation appears to have a hidden or stale "
            "GPU context that is blocking a non-destructive research sweep.",
            *common,
            f"GPU: {gpu.get('name', 'unknown')}",
            f"Memory used: {memory_used}/{memory_total} MiB",
            f"Utilization: {utilization}%",
            f"Visible compute apps: {_count(gpu.get('compute_apps'))}",
            f"Accounted apps: {_count(gpu.get('accounted_apps'))}",
            f"Local /proc/*/fd NVIDIA holders: {_count(gpu.get('device_holders'))}",
            f"nvidia-smi PIDS: {_pids_summary(gpu.get('pid_query'))}",
            (
                "The project launcher is intentionally waiting for memory "
                f"<= {memory_threshold} MiB and utilization "
                f"<= {utilization_threshold}%."
            ),
            "Please release/restart the notebook allocation or clear the stale "
            "GPU context from the infrastructure side.",
            "We have not run nvidia-smi --gpu-reset or killed unknown processes.",
        ]
    else:
        body = [
            "Hello, I need help triaging an H200 notebook GPU allocation before "
            "starting a non-destructive research sweep.",
            *common,
            f"GPU: {gpu.get('name', 'unknown')}",
            f"Memory used: {memory_used}/{memory_total} MiB",
            f"Utilization: {utilization}%",
            f"Visible GPU holders: {visible_holder_count}",
            f"Hidden GPU context likely: {_bool(status.get('hidden_gpu_context_likely'))}",
            "The attached diagnostic report should not be treated as hidden-context "
            "evidence while visible holders are present or the experiment/launcher "
            "state is not safe for that conclusion.",
            "Please help identify the visible GPU process or device-holder "
            "ownership before any restart or cleanup action.",
            "We have not run nvidia-smi --gpu-reset or killed unknown processes.",
        ]
    if stale_note:
        body.append(stale_note)
    return body


def _requested_non_destructive_action(
    status: dict[str, Any], gpu: dict[str, Any]
) -> list[str]:
    if _hidden_context_case(status, gpu):
        action = (
            "Please release or restart the notebook allocation, or clear the stale "
            "GPU context from the infrastructure side. The project launcher is "
            "intentionally waiting for low GPU memory and utilization before "
            "starting the registered sweep. We have not run `nvidia-smi "
            "--gpu-reset` or killed unknown processes."
        )
        rerun_label = "After the allocation is cleared, rerun:"
    elif not gpu.get("available"):
        action = (
            "Please verify that the H200 allocation is attached and visible to the "
            "notebook/runtime. We have not run `nvidia-smi --gpu-reset` or killed "
            "unknown processes."
        )
        rerun_label = "After GPU visibility is restored, rerun:"
    else:
        action = (
            "Please identify the visible GPU process or device-holder ownership "
            "before any cleanup action. This report should not be used as hidden "
            "GPU-context evidence while visible holders are present or an "
            "experiment appears to be running. We have not run `nvidia-smi "
            "--gpu-reset` or killed unknown processes."
        )
        rerun_label = "After the allocation state is clarified, rerun:"

    return [
        "",
        "## Requested Non-Destructive Action",
        "",
        action,
        "",
        rerun_label,
        "",
        "```bash",
        "cd /home/aryang9/sandbox/llm-safety",
        "uv run python scripts/report_h200_status.py",
        "```",
        "",
    ]


def _hidden_context_case(status: dict[str, Any], gpu: dict[str, Any]) -> bool:
    return (
        bool(gpu.get("available"))
        and bool(status.get("hidden_gpu_context_likely"))
        and bool(status.get("launcher_waiting"))
        and not bool(status.get("experiment_running"))
        and _visible_holder_count(gpu) == 0
    )


def _visible_holder_count(gpu: dict[str, Any]) -> int:
    return (
        _count(gpu.get("compute_apps"))
        + _count(gpu.get("accounted_apps"))
        + _count(gpu.get("device_holders"))
    )


def _launcher_stale_note(status: dict[str, Any]) -> str:
    wait_history = status.get("launcher_log", {}).get("wait_history") or {}
    if wait_history.get("launcher_log_stale"):
        return (
            "The launcher log marks the latest sample as stale; please rerun the "
            "status command before acting if the notebook state may have changed."
        )
    return ""


def _pids_summary(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return "None" if "Processes" in text and "None" in text else "see diagnostic report"


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def _count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _reasons(value: Any) -> str:
    if isinstance(value, list) and value:
        return ", ".join(str(item) for item in value)
    return "none"


if __name__ == "__main__":
    main()
