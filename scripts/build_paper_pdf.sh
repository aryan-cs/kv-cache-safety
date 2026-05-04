#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

src_dir="paper/latex"
build_dir="paper/build"
mkdir -p "$build_dir"

if [[ "${REQUIRE_COMPLETE_PAPER:-0}" == "1" ]]; then
  uv run python scripts/check_latex_placeholders.py --tex "$src_dir/main.tex"
fi

if command -v tectonic >/dev/null 2>&1; then
  (
    cd "$src_dir"
    tectonic --outdir ../build main.tex
  )
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" "$src_dir/main.tex"
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  (
    cd "$src_dir"
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
    bibtex ../build/main
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
  )
else
  echo "No supported LaTeX builder found. Install tectonic, latexmk, or pdflatex+bibtex." >&2
  exit 1
fi

if [[ -f "$build_dir/main.pdf" ]]; then
  mv "$build_dir/main.pdf" "$build_dir/cache_mediated_safety_erasure.pdf"
fi

echo "Wrote $build_dir/cache_mediated_safety_erasure.pdf"
