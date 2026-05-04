from __future__ import annotations

import argparse
import re
from pathlib import Path

PLACEHOLDER_PATTERNS = [
    re.compile(r"\\maybeincludegraphic\{([^{}]+)\}"),
    re.compile(r"\\maybeinputtable\{([^{}]+)\}"),
    re.compile(r"\\requiredartifact\{([^{}]+)\}"),
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

    missing = missing_placeholder_artifacts(args.tex)
    if missing:
        print("LATEX PLACEHOLDER CHECK FAILED")
        for path in missing:
            print(f"- missing artifact: {path}")
        raise SystemExit(1)
    print("LATEX PLACEHOLDER CHECK PASSED")


def missing_placeholder_artifacts(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    base_dir = tex_path.parent
    missing: list[str] = []
    for pattern in PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(text):
            raw_path = match.group(1)
            path = (base_dir / raw_path).resolve()
            if not path.exists():
                missing.append(raw_path)
    return sorted(set(missing))


if __name__ == "__main__":
    main()
