from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.judging.local_cli import JudgeCommand, judge_row, judgment_key
from cache_safety_erasure.utils.io import read_jsonl_tolerant


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Judge generated rows locally with Codex/GPT-5.5 and Gemini fallback. "
            "Outputs are source-marked model-judge labels, not human labels."
        )
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--providers", default="codex,gemini")
    parser.add_argument("--codex-model", default="gpt-5.5")
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
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
        done = {str(row.get("judgment_key")) for row in existing if row.get("judgment_key")}
    pending = [row for row in rows if judgment_key(row) not in done]
    commands = _commands_from_args(args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = len(done)
    total = len(done) + len(pending)
    print(f"Judging {len(pending)} pending row(s); {len(done)} already complete.")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(judge_row, row, commands): row for row in pending}
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


if __name__ == "__main__":
    main()
