from __future__ import annotations

import argparse
import re
from pathlib import Path

from check_final_pdf_text import forbidden_final_prose_failures
from check_publication_readiness import _pdf_stream_visual_failure

PLACEHOLDER_PATTERNS = [
    re.compile(r"\\maybeincludegraphic\{([^{}]+)\}"),
    re.compile(r"\\maybeinputtable\{([^{}]+)\}"),
    re.compile(r"\\requiredartifact\{([^{}]+)\}"),
]
PLACEHOLDER_TEXT_MARKERS = [
    "Empirical result not yet reported",
    "Figure pending",
    "Result pending",
    "Results pending; no readiness-passing rows exported.",
    "results pending",
    "This draft",
    "draft manuscript",
    "must replace these placeholders",
]
REQUIRED_TEX_MARKERS_BY_NAME = {
    "main_results_table.tex": [
        "policy level ssei",
        "policy level ssei ci low",
        "policy level ssei ci high",
    ],
    "suite_level_effects_table.tex": [
        "paired n",
        "cluster n",
        "safety ci low",
        "safety ci high",
    ],
    "causal_restoration_table.tex": [
        "safety ci low",
        "safety ci high",
        "refusal ci low",
        "refusal ci high",
    ],
}
REQUIRED_RESULT_MACROS = [
    (
        ("h200_qwen_full_sweep", "active_primary"),
        [
            "PrimaryRunId",
            "PrimaryPolicyCount",
            "PrimaryTopSSEIPolicy",
            "PrimaryTopSSEI",
            "PrimaryTopSSEICILow",
            "PrimaryTopSSEICIHigh",
            "PrimarySafetyClusterCount",
            "PrimaryCapabilityClusterCount",
        ],
    ),
    (
        ("h200_causal_patch_qwen7b", "active_causal"),
        [
            "CausalRunId",
            "CausalPolicyCount",
        ],
    ),
]
REQUIRED_CAUSAL_ROW_MARKERS = [
    "rolesystem",
    "roleuser",
    "policy_pinned",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if the LaTeX manuscript would render placeholder figures, tables, "
            "or required generated text."
        )
    )
    parser.add_argument("--tex", type=Path, default=Path("paper/latex/main.tex"))
    args = parser.parse_args()

    failures = placeholder_artifact_failures(args.tex)
    if failures:
        print("LATEX PLACEHOLDER CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("LATEX PLACEHOLDER CHECK PASSED")


def missing_placeholder_artifacts(tex_path: Path) -> list[str]:
    return sorted(
        {
            raw_path
            for raw_path, path in _placeholder_artifacts(tex_path)
            if not path.exists()
        }
    )


def placeholder_artifact_failures(tex_path: Path) -> list[str]:
    failures: list[str] = []
    for raw_path, path in _placeholder_artifacts(tex_path):
        if not path.exists():
            failures.append(f"missing artifact: {raw_path}")
            continue
        if path.suffix.lower() == ".pdf" and not _is_pdf(path):
            failures.append(f"invalid PDF artifact: {raw_path}")
        if path.suffix.lower() in {".tex", ".md", ".csv"}:
            if path.stat().st_size == 0:
                failures.append(f"empty artifact: {raw_path}")
                continue
            if path.suffix.lower() == ".tex":
                text = path.read_text(encoding="utf-8", errors="replace")
                rendered_text = _strip_tex_comments(text)
                text_lower = text.lower()
                for marker in PLACEHOLDER_TEXT_MARKERS:
                    if marker.lower() in text_lower:
                        failures.append(f"placeholder text in artifact: {raw_path}")
                        break
                failures.extend(
                    f"forbidden final prose in artifact: {raw_path}::{failure}"
                    for failure in forbidden_final_prose_failures(rendered_text)
                )
                failures.extend(_semantic_tex_failures(raw_path, path.name, text))
    return sorted(set(failures))


def _placeholder_artifacts(tex_path: Path) -> list[tuple[str, Path]]:
    text = tex_path.read_text(encoding="utf-8")
    base_dir = tex_path.parent
    artifacts: list[tuple[str, Path]] = []
    for pattern in PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(text):
            raw_path = match.group(1)
            path = (base_dir / raw_path).resolve()
            artifacts.append((raw_path, path))
    return artifacts


def _is_pdf(path: Path) -> bool:
    try:
        content = path.read_bytes()
    except OSError:
        return False
    if not (
        content.startswith(b"%PDF-")
        and len(content) >= 32
        and b"%%EOF" in content[-2048:]
    ):
        return False
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if not reader.pages:
            return False
        for page in reader.pages:
            if float(page.mediabox.width) <= 32 or float(page.mediabox.height) <= 32:
                return False
            content_stream = page.get_contents()
            if content_stream is None:
                return False
            try:
                stream_data = content_stream.get_data()
            except AttributeError:
                stream_data = b"".join(stream.get_data() for stream in content_stream)
            if len(stream_data.strip()) < 32:
                return False
            if _pdf_stream_visual_failure(stream_data, 1):
                return False
    except Exception:
        return False
    return True


def _strip_tex_comments(text: str) -> str:
    rendered_lines = []
    for line in text.splitlines():
        rendered_lines.append(_strip_tex_comment_line(line))
    return "\n".join(rendered_lines)


def _strip_tex_comment_line(line: str) -> str:
    escaped = False
    for index, char in enumerate(line):
        if char == "%" and not escaped:
            return line[:index]
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    return line


def _semantic_tex_failures(raw_path: str, name: str, text: str) -> list[str]:
    failures: list[str] = []
    normalized = text.lower().replace(r"\_", "_")
    for marker in REQUIRED_TEX_MARKERS_BY_NAME.get(name, []):
        if marker not in normalized:
            failures.append(f"missing required table marker in artifact: {raw_path}::{marker}")
    if name == "causal_restoration_table.tex":
        for marker in REQUIRED_CAUSAL_ROW_MARKERS:
            if marker not in normalized:
                failures.append(f"missing causal control row in artifact: {raw_path}::{marker}")
    if name == "result_macros.tex":
        macro_values = _macro_values(text)
        for path_markers, required_macros in REQUIRED_RESULT_MACROS:
            if not any(path_marker in raw_path for path_marker in path_markers):
                continue
            for macro in required_macros:
                value = macro_values.get(macro, "").strip()
                if not value:
                    failures.append(f"missing required macro in artifact: {raw_path}::{macro}")
    return failures


def _macro_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in re.finditer(r"\\renewcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}", text):
        values[match.group(1)] = match.group(2)
    return values


if __name__ == "__main__":
    main()
