from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgePaths:
    audit_csv: Path
    audit_key: Path
    approved_input: Path
    judgments: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a completed H200 selectivity run and run local Codex/Gemini audit "
            "judging without writing judge state into the H200-owned results directory."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-run-dir", default=None)
    parser.add_argument("--local-results-dir", type=Path, default=Path("results"))
    parser.add_argument("--audit-dir", type=Path, default=Path("docs/audit"))
    parser.add_argument("--per-suite-policy", type=int, default=8)
    parser.add_argument("--strategy", choices=["effect", "random"], default="effect")
    parser.add_argument("--providers", default="codex,gemini")
    parser.add_argument("--judge-mode", choices=["all-providers", "first-success"], default="all-providers")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--approval-note",
        default="User approved local Codex/Gemini judging for H200-generated selectivity audit rows.",
    )
    args = parser.parse_args()

    run_dir = args.local_results_dir / args.run_id
    remote_run_dir = args.remote_run_dir or f"results/{args.run_id}"
    paths = judge_paths_for_run(args.run_id, args.audit_dir)

    commands: list[list[str]] = []
    if not args.skip_fetch:
        commands.append(
            [
                "bash",
                "scripts/fetch_h200_selectivity_results.sh",
                remote_run_dir,
                str(args.local_results_dir),
            ]
        )

    if commands:
        _run_commands(commands, dry_run=args.dry_run)

    if args.dry_run:
        _print_planned_judging(args, run_dir, paths)
        return

    if not run_dir.exists():
        raise SystemExit(f"Fetched run directory not found: {run_dir}")
    complete, reason = is_run_complete(run_dir)
    if not complete and not args.allow_partial:
        print(f"Skipping judging for incomplete run {args.run_id}: {reason}")
        return

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    _run_commands(
        [
            [
                sys.executable,
                "scripts/export_human_audit_sample.py",
                "--results-dir",
                str(run_dir),
                "--output-dir",
                str(args.audit_dir),
                "--per-suite-policy",
                str(args.per_suite_policy),
                "--strategy",
                args.strategy,
                "--include-hidden-reference",
            ],
            [
                sys.executable,
                "scripts/approve_judge_egress.py",
                "--input-jsonl",
                str(paths.audit_key),
                "--output-jsonl",
                str(paths.approved_input),
                "--approval-note",
                args.approval_note,
                "--approval-source",
                "user_instruction",
                "--overwrite",
            ],
            _judge_command(args, paths),
        ],
        dry_run=False,
    )


def judge_paths_for_run(run_id: str, audit_dir: Path) -> JudgePaths:
    return JudgePaths(
        audit_csv=audit_dir / f"{run_id}_audit_blinded.csv",
        audit_key=audit_dir / f"{run_id}_audit_key.jsonl",
        approved_input=audit_dir / f"{run_id}_audit_key.codex_gemini_approved.jsonl",
        judgments=audit_dir / f"{run_id}_judgments.codex_gemini.jsonl",
    )


def is_run_complete(run_dir: Path) -> tuple[bool, str]:
    progress = _read_json(run_dir / "progress.json")
    manifest = _read_json(run_dir / "manifest.json")
    generations_path = run_dir / "generations.jsonl"
    expected = int(progress.get("expected") or manifest.get("expected_generation_count") or 0)
    completed = int(progress.get("completed") or _jsonl_row_count(generations_path))
    activity = str(progress.get("activity") or "")
    if expected <= 0:
        return False, "missing expected row count"
    if completed < expected:
        return False, f"{completed}/{expected} rows complete"
    if activity and activity != "complete":
        return False, f"progress activity is {activity!r}"
    return True, f"{completed}/{expected} rows complete"


def _judge_command(args: argparse.Namespace, paths: JudgePaths) -> list[str]:
    command = [
        sys.executable,
        "scripts/judge_with_codex_gemini.py",
        "--input-jsonl",
        str(paths.approved_input),
        "--output-jsonl",
        str(paths.judgments),
        "--providers",
        args.providers,
        "--judge-mode",
        args.judge_mode,
        "--workers",
        str(args.workers),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--resume",
        "--allow-data-egress",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def _run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        printable = " ".join(command)
        if dry_run:
            print(printable)
            continue
        print(f"+ {printable}")
        subprocess.run(command, check=True)


def _print_planned_judging(args: argparse.Namespace, run_dir: Path, paths: JudgePaths) -> None:
    print(f"local run dir: {run_dir}")
    print(f"audit csv: {paths.audit_csv}")
    print(f"audit key: {paths.audit_key}")
    print(f"approved judge input: {paths.approved_input}")
    print(f"judgments: {paths.judgments}")
    print("planned local commands:")
    _run_commands(
        [
            [
                sys.executable,
                "scripts/export_human_audit_sample.py",
                "--results-dir",
                str(run_dir),
                "--output-dir",
                str(args.audit_dir),
                "--per-suite-policy",
                str(args.per_suite_policy),
                "--strategy",
                args.strategy,
                "--include-hidden-reference",
            ],
            [
                sys.executable,
                "scripts/approve_judge_egress.py",
                "--input-jsonl",
                str(paths.audit_key),
                "--output-jsonl",
                str(paths.approved_input),
                "--approval-note",
                args.approval_note,
                "--approval-source",
                "user_instruction",
                "--overwrite",
            ],
            _judge_command(args, paths),
        ],
        dry_run=True,
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
