from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.judging.local_cli import (
    PROMPT_PROTOCOL_VERSION,
    JudgeCommand,
    judge_row,
    judgment_key,
)
from cache_safety_erasure.utils.io import read_jsonl_tolerant


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Judge generated rows locally with Codex/GPT-5.4 and Gemini fallback. "
            "Outputs are source-marked model-judge labels, not human labels."
        )
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--providers", default="codex,gemini")
    parser.add_argument(
        "--judge-mode",
        choices=["all-providers", "first-success"],
        default="all-providers",
        help=(
            "all-providers writes one source-marked row per eligible judge for disagreement "
            "analysis; first-success keeps the historical fallback behavior."
        ),
    )
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-statuses",
        default="",
        help=(
            "Comma-separated parser_status values to retry even when --resume is set. "
            "Use this for transient judge failures such as blocked,parse_error,unlabeled. "
            "Existing attempts are preserved; successful retries are appended with the same "
            "source-marked judgment key."
        ),
    )
    parser.add_argument(
        "--allow-data-egress",
        action="store_true",
        help=(
            "Permit external/proprietary judge calls. Rows must also have the data-egress "
            "approval field set truthy."
        ),
    )
    parser.add_argument(
        "--data-egress-field",
        default="data_egress_approved",
        help="Input row field that must be truthy before external/proprietary judging.",
    )
    args = parser.parse_args()

    rows, corrupt_input = read_jsonl_tolerant(args.input_jsonl)
    if corrupt_input is not None:
        print(f"Quarantined corrupt input tail at {corrupt_input}.")
    if args.limit is not None:
        rows = rows[: args.limit]

    done = set()
    if args.resume and args.output_jsonl.exists():
        existing, corrupt_output = read_jsonl_tolerant(args.output_jsonl)
        if corrupt_output is not None:
            print(f"Quarantined corrupt output tail at {corrupt_output}.")
        retry_statuses = _retry_statuses_from_args(args.retry_statuses)
        done = {
            _done_key(row, mode=args.judge_mode)
            for row in existing
            if row.get("judgment_key")
            and str(row.get("parser_status", "")).strip() not in retry_statuses
        }
    commands = _commands_from_args(args)
    tasks = _judging_tasks(rows, commands, done, mode=args.judge_mode)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = len(done)
    total = len(done) + len(tasks)
    print(f"Judging {len(tasks)} pending task(s); {len(done)} already complete.")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                judge_row,
                task["row"],
                task["commands"],
                allow_data_egress=args.allow_data_egress,
                data_egress_field=args.data_egress_field,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            judgment = future.result()
            with lock:
                with args.output_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(judgment, sort_keys=True) + "\n")
                completed += 1
                percent = (completed / total) * 100 if total else 100.0
                print(
                    f"[local_judging] Progress: {percent:.1f}% "
                    f"({completed}/{total}) {judgment.get('judgment_key')}"
                )


def _commands_from_args(args: argparse.Namespace) -> list[JudgeCommand]:
    providers = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    commands: list[JudgeCommand] = []
    for provider in providers:
        if provider == "codex":
            commands.append(
                JudgeCommand(
                    provider="codex",
                    model=args.codex_model,
                    timeout_seconds=args.timeout_seconds,
                    cwd=Path.cwd(),
                )
            )
        elif provider == "gemini":
            commands.append(
                JudgeCommand(
                    provider="gemini",
                    model=args.gemini_model,
                    timeout_seconds=args.timeout_seconds,
                    cwd=Path.cwd(),
                )
            )
        else:
            raise ValueError(f"Unsupported provider in --providers: {provider}")
    if not commands:
        raise ValueError("--providers must include at least one provider")
    return commands


def _judging_tasks(
    rows: list[dict],
    commands: list[JudgeCommand],
    done: set[str],
    *,
    mode: str,
) -> list[dict]:
    if mode == "first-success":
        tasks = []
        seen: set[str] = set()
        for row in rows:
            key = _first_success_done_key(row)
            if key in done or key in seen:
                continue
            seen.add(key)
            tasks.append({"row": row, "commands": commands})
        return tasks
    tasks = []
    seen = set()
    for row in rows:
        for command in commands:
            key = _provider_done_key(row, command)
            if key in done or key in seen:
                continue
            seen.add(key)
            tasks.append({"row": row, "commands": [command]})
    return tasks


def _done_key(row: dict, *, mode: str) -> str:
    protocol = str(row.get("judge_prompt_protocol_version", ""))
    if mode == "first-success":
        return "::".join([str(row.get("judgment_key")), protocol])
    return "::".join(
        [
            str(row.get("judgment_key")),
            str(row.get("judge_provider", "")),
            str(row.get("judge_model", "")),
            protocol,
        ]
    )


def _retry_statuses_from_args(value: str) -> set[str]:
    return {status.strip() for status in value.split(",") if status.strip()}


def _provider_done_key(row: dict, command: JudgeCommand) -> str:
    return "::".join([judgment_key(row), command.provider, command.model or "", PROMPT_PROTOCOL_VERSION])


def _first_success_done_key(row: dict) -> str:
    return "::".join([judgment_key(row), PROMPT_PROTOCOL_VERSION])


if __name__ == "__main__":
    main()
