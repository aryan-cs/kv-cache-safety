from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl_tolerant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print heartbeat-friendly status for a restartable selectivity run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--activity", default=None)
    parser.add_argument("--paper-purpose", default=None)
    args = parser.parse_args()

    status = build_status(args.run_dir, activity_override=args.activity)
    purpose = args.paper_purpose or (
        "This run measures whether cache pressure changes safety behavior more than "
        "matched capability behavior."
    )
    print(
        f"[{status['activity']}] Progress: {status['progress_percent']:.1f}%\n"
        f"{purpose}\n"
        f"Currently {status['verb']} {status['current_label']}\n"
        f"Estimated Time Remaining: {status['eta_minutes']} minutes"
    )


def build_status(run_dir: Path, *, activity_override: str | None = None) -> dict[str, Any]:
    progress_path = run_dir / "progress.json"
    progress = _read_json(progress_path)
    generations, _ = read_jsonl_tolerant(run_dir / "generations.jsonl")
    manifest = _read_json(run_dir / "manifest.json")

    expected = int(progress.get("expected") or manifest.get("expected_generation_count") or 0)
    completed = int(progress.get("completed") or len(generations))
    percent = (completed / expected) * 100 if expected else (100.0 if completed else 0.0)
    activity = activity_override or str(progress.get("activity") or "idle")
    current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    current_label = _current_label(current, manifest)
    eta_minutes = _estimate_eta_minutes(generations, completed=completed, expected=expected)
    verb = "running" if activity not in {"complete", "idle"} else activity
    return {
        "activity": activity,
        "progress_percent": percent,
        "completed": completed,
        "expected": expected,
        "current_label": current_label,
        "eta_minutes": eta_minutes,
        "verb": verb,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _current_label(current: dict[str, Any], manifest: dict[str, Any]) -> str:
    model_id = current.get("model_id") or manifest.get("model_id") or "unknown model"
    suite = current.get("suite")
    policy = current.get("policy")
    prompt_id = current.get("prompt_id")
    parts = [str(model_id)]
    if suite:
        parts.append(str(suite))
    if policy:
        parts.append(str(policy))
    if prompt_id:
        parts.append(str(prompt_id))
    return " / ".join(parts)


def _estimate_eta_minutes(
    rows: list[dict[str, Any]], *, completed: int, expected: int
) -> int | str:
    remaining = max(0, expected - completed)
    if remaining == 0:
        return 0
    timestamps = [
        _parse_timestamp(str(row.get("generated_at", "")))
        for row in rows
        if row.get("generated_at")
    ]
    timestamps = [ts for ts in timestamps if ts is not None]
    if len(timestamps) < 2:
        return "unknown"
    elapsed = (max(timestamps) - min(timestamps)).total_seconds()
    if elapsed <= 0:
        return "unknown"
    rate = (len(timestamps) - 1) / elapsed
    if rate <= 0:
        return "unknown"
    return int(round((remaining / rate) / 60))


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


if __name__ == "__main__":
    main()
