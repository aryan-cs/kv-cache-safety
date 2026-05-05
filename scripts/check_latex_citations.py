from __future__ import annotations

import argparse
import re
from pathlib import Path

CITE_PATTERN = re.compile(r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}")
BIB_KEY_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
ENTRY_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=^@\w+\s*\{|\Z)", re.DOTALL | re.MULTILINE)
FIELD_PATTERN = re.compile(r"(\w+)\s*=\s*(?:\{([^{}]*)\}|\"([^\"]*)\"|([^,\n]+))", re.DOTALL)
PLACEHOLDER_VALUE_PATTERN = re.compile(
    r"\b(?:tbd|todo|placeholder|citation needed|unknown)\b",
    re.IGNORECASE,
)
REQUIRED_SUPPORT_FIELDS = {"url", "doi", "eprint", "journal", "booktitle", "howpublished", "note"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail on unresolved, duplicate, or optionally unused LaTeX citations."
    )
    parser.add_argument("--tex", required=True, type=Path)
    parser.add_argument("--bib", required=True, type=Path)
    parser.add_argument("--require-all-bib-used", action="store_true")
    args = parser.parse_args()

    failures = citation_failures(
        args.tex,
        args.bib,
        require_all_bib_used=args.require_all_bib_used,
    )
    if failures:
        print("LATEX CITATION CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("LATEX CITATION CHECK PASSED")


def citation_failures(
    tex_path: Path, bib_path: Path, *, require_all_bib_used: bool = False
) -> list[str]:
    failures: list[str] = []
    try:
        tex = tex_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"missing_or_unreadable_tex:{tex_path}:{exc}"]
    try:
        bib = bib_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"missing_or_unreadable_bib:{bib_path}:{exc}"]

    rendered_tex = _strip_tex_comments(tex)
    cited_keys = _citation_keys(rendered_tex)
    bib_keys = _bib_keys(bib)
    bib_entries = _bib_entries(bib)
    bib_key_set = set(bib_keys)
    duplicate_bib_keys = sorted({key for key in bib_keys if bib_keys.count(key) > 1})
    for key in duplicate_bib_keys:
        failures.append(f"duplicate_bib_key:{key}")
    for key in sorted(cited_keys - bib_key_set):
        failures.append(f"missing_bib_entry:{key}")
    for key in sorted(cited_keys & bib_key_set):
        failures.extend(_bib_entry_quality_failures(key, bib_entries.get(key, {})))
    if require_all_bib_used:
        for key in sorted(bib_key_set - cited_keys):
            failures.append(f"unused_bib_entry:{key}")
    return failures


def _citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in CITE_PATTERN.finditer(tex):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _bib_keys(bib: str) -> list[str]:
    return [match.group(1).strip() for match in BIB_KEY_PATTERN.finditer(bib)]


def _bib_entries(bib: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for key, body in ENTRY_PATTERN.findall(bib):
        fields = {}
        for name, braced, quoted, bare in FIELD_PATTERN.findall(body):
            fields[name.lower()] = (braced or quoted or bare).strip().rstrip(",")
        entries[key.strip()] = fields
    return entries


def _bib_entry_quality_failures(key: str, fields: dict[str, str]) -> list[str]:
    failures = []
    for required in ["title", "year"]:
        value = fields.get(required, "").strip()
        if not value:
            failures.append(f"bib_entry_missing_field:{key}:{required}")
        elif PLACEHOLDER_VALUE_PATTERN.search(value):
            failures.append(f"bib_entry_placeholder_field:{key}:{required}")
    if not any(fields.get(name, "").strip() for name in REQUIRED_SUPPORT_FIELDS):
        failures.append(f"bib_entry_lacks_support_locator:{key}")
    for name, value in fields.items():
        if PLACEHOLDER_VALUE_PATTERN.search(value):
            failures.append(f"bib_entry_placeholder_field:{key}:{name}")
    return sorted(set(failures))


def _strip_tex_comments(text: str) -> str:
    return "\n".join(_strip_tex_comment_line(line) for line in text.splitlines())


def _strip_tex_comment_line(line: str) -> str:
    escaped = False
    for index, char in enumerate(line):
        if char == "%" and not escaped:
            return line[:index]
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    return line


if __name__ == "__main__":
    main()
