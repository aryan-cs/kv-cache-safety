from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl_tolerant, utc_timestamp, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a judge-input JSONL with explicit data-egress approval provenance."
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--approval-source", default="user_instruction")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_jsonl.exists() and not args.overwrite:
        raise SystemExit(f"{args.output_jsonl} exists; pass --overwrite to replace it.")

    rows, corrupt_input = read_jsonl_tolerant(args.input_jsonl)
    if corrupt_input is not None:
        print(f"Quarantined corrupt input tail at {corrupt_input}.")
    write_jsonl(
        args.output_jsonl,
        approve_rows(
            rows,
            approval_note=args.approval_note,
            approval_source=args.approval_source,
            approved_at=utc_timestamp(),
        ),
    )
    print(f"Wrote {len(rows)} approved judge-input row(s) to {args.output_jsonl}")


def approve_rows(
    rows: list[dict],
    *,
    approval_note: str,
    approval_source: str,
    approved_at: str,
) -> list[dict]:
    return [
        {
            **row,
            "data_egress_approved": True,
            "data_egress_approval_source": approval_source,
            "data_egress_approval_note": approval_note,
            "data_egress_approved_at": approved_at,
        }
        for row in rows
    ]


if __name__ == "__main__":
    main()
