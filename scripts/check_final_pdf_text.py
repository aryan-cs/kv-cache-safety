from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PLACEHOLDER_MARKERS = [
    "Empirical result not yet reported",
    "Figure unavailable",
    "Figure pending",
    "Result pending",
    "Results pending; no readiness-passing rows exported.",
    "results pending",
    "registered analysis protocol",
    "reports no empirical claims",
    "This draft",
    "draft manuscript",
    "must replace these placeholders",
]
FORBIDDEN_FINAL_PROSE_PATTERNS: list[tuple[str, str]] = [
    ("H200", r"\bh\s*200\b"),
    ("MacBook", r"\bmac\s*book\b"),
    ("cgroup", r"\bc\s*group\b"),
    ("nvidia-smi", r"\bnvidia\s*smi\b"),
    ("Illinois Computes", r"\billinois\s+computes\b"),
    ("141GB allocation", r"\b141\s*g\s*b\b"),
    ("32GB RAM allocation", r"\b32\s*g\s*b\s+r\s*a\s*m\b"),
    ("hardware constraint", r"\bhardware\s+constraint\b"),
    ("notebook allocation", r"\bnotebook\s+allocation\b"),
    ("support bundle", r"\bsupport\s+bundle\b"),
    ("infrastructure diagnostics", r"\binfrastructure\s+diagnostics\b"),
    ("visible compute apps", r"\bvisible\s+compute\s+apps?\b"),
    ("visible compute process", r"\bvisible\s+compute\s+process(?:es)?\b"),
    ("admin report", r"\badmin\s+report\b"),
    ("smoke run", r"\bsmoke\s+run\b"),
    ("mock model", r"\bmock\s+model\b"),
    ("dirty tree", r"\bdirty\s+(?:git\s+)?(?:working\s+)?tree\b"),
    ("pre-results", r"\bpre\s+results\b"),
    ("draft-only", r"\bdraft\s+only\b"),
    ("not a publishable paper", r"\bnot\s+a\s+publishable\s+paper\b"),
    ("evidence-gated fallback", r"\bevidence\s+gated\s+fallback\b"),
    ("launcher", r"\blauncher\b"),
    ("finalizer", r"\bfinalizer\b"),
]
RESOURCE_STATUS_WORDS = (
    r"available|availability|busy|free|used|instance|queue|queued|gate|gated|"
    r"wait|waiting|running|concurrently|process|status|diagnostic|diagnostics|"
    r"notebook|support|visible|admin|launcher|finalizer"
)
FORBIDDEN_RESOURCE_CONTEXT_PATTERNS: list[tuple[str, str]] = [
    (
        "GPU operational status",
        rf"\bg\s*p\s*u\b(?:\s+\w+){{0,12}}\s+(?:{RESOURCE_STATUS_WORDS})\b"
        rf"|\b(?:{RESOURCE_STATUS_WORDS})\b(?:\s+\w+){{0,12}}\s+\bg\s*p\s*u\b",
    ),
    (
        "CUDA operational status",
        rf"\bc\s*u\s*d\s*a\b(?:\s+\w+){{0,12}}\s+(?:{RESOURCE_STATUS_WORDS})\b"
        rf"|\b(?:{RESOURCE_STATUS_WORDS})\b(?:\s+\w+){{0,12}}\s+\bc\s*u\s*d\s*a\b",
    ),
    (
        "VRAM operational status",
        rf"\bv\s*r\s*a\s*m\b(?:\s+\w+){{0,12}}\s+(?:{RESOURCE_STATUS_WORDS})\b"
        rf"|\b(?:{RESOURCE_STATUS_WORDS})\b(?:\s+\w+){{0,12}}\s+\bv\s*r\s*a\s*m\b",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if the final rendered PDF still contains draft/protocol placeholders."
    )
    parser.add_argument("--pdf", required=True, type=Path)
    args = parser.parse_args()

    text, extractor = extract_pdf_text(args.pdf)
    failures = final_pdf_text_failures(text, extractor)
    if failures:
        print("FINAL PDF TEXT CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"FINAL PDF TEXT CHECK PASSED ({extractor})")


def placeholder_text_failures(text: str) -> list[str]:
    text_lower = text.lower()
    failures = [
        f"placeholder_text:{marker}"
        for marker in PLACEHOLDER_MARKERS
        if marker.lower() in text_lower
    ]
    failures.extend(forbidden_final_prose_failures(text))
    return failures


def final_pdf_text_failures(text: str, extractor: str) -> list[str]:
    failures = placeholder_text_failures(text)
    if not text.strip():
        failures.append(f"empty_extracted_pdf_text:{extractor}")
    return failures


def forbidden_final_prose_failures(text: str) -> list[str]:
    normalized = _normalize_final_prose(text)
    failures = [
        f"forbidden_final_prose:{marker}"
        for marker, pattern in FORBIDDEN_FINAL_PROSE_PATTERNS
        if re.search(pattern, normalized)
    ]
    failures.extend(
        f"forbidden_final_prose:{marker}"
        for marker, pattern in FORBIDDEN_RESOURCE_CONTEXT_PATTERNS
        if re.search(pattern, normalized)
    )
    return failures


def _normalize_final_prose(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    normalized = normalized.replace("\\", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


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
