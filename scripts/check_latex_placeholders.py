from __future__ import annotations

import argparse
import re
from pathlib import Path

PLACEHOLDER_PATTERNS = [
    re.compile(r"\\maybeincludegraphic\{([^{}]+)\}"),
    re.compile(r"\\maybeinputtable\{([^{}]+)\}"),
    re.compile(r"\\requiredartifact\{([^{}]+)\}"),
]
PLACEHOLDER_TEXT_MARKERS = [
    "Empirical result not yet reported",
    "Results pending; no readiness-passing rows exported.",
    "results pending",
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
                text_lower = text.lower()
                for marker in PLACEHOLDER_TEXT_MARKERS:
                    if marker.lower() in text_lower:
                        failures.append(f"placeholder text in artifact: {raw_path}")
                        break
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
        return path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


if __name__ == "__main__":
    main()
