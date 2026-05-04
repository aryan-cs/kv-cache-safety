from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

PLACEHOLDER_MARKERS = [
    "Empirical result not yet reported",
    "Figure unavailable",
    "Results pending; no readiness-passing rows exported.",
    "results pending",
    "registered analysis protocol",
    "reports no empirical claims",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if the final rendered PDF still contains draft/protocol placeholders."
    )
    parser.add_argument("--pdf", required=True, type=Path)
    args = parser.parse_args()

    text, extractor = extract_pdf_text(args.pdf)
    failures = placeholder_text_failures(text)
    if not text.strip():
        failures.append(f"empty_extracted_pdf_text:{extractor}")
    if failures:
        print("FINAL PDF TEXT CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"FINAL PDF TEXT CHECK PASSED ({extractor})")


def placeholder_text_failures(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        f"placeholder_text:{marker}"
        for marker in PLACEHOLDER_MARKERS
        if marker.lower() in text_lower
    ]


def extract_pdf_text(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "pypdf"
    except Exception as pypdf_error:
        if shutil.which("pdftotext"):
            with tempfile.NamedTemporaryFile(suffix=".txt") as output:
                completed = subprocess.run(
                    ["pdftotext", str(path), output.name],
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if completed.returncode == 0:
                    return Path(output.name).read_text(encoding="utf-8", errors="replace"), (
                        "pdftotext"
                    )
                raise RuntimeError(completed.stderr.strip() or "pdftotext failed") from pypdf_error
        raise RuntimeError(
            "Could not extract final PDF text with pypdf, and pdftotext is unavailable: "
            f"{pypdf_error}"
        ) from pypdf_error


if __name__ == "__main__":
    main()
