from pathlib import Path


def test_latex_manuscript_is_arxiv_friendly_pre_results_draft() -> None:
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[11pt]{article}" in tex
    assert "Aryan Gupta" in tex
    assert "aryan.cs.app@gmail.com" in tex
    assert "pre-results" in tex.lower()
    assert "reports no empirical claims" in tex
    assert "Result pending" in tex
    assert r"\bibliography{../references}" in tex
    assert "neurips" not in tex.lower()


def test_latex_references_cover_primary_model_and_cache_work() -> None:
    bib = Path("paper/references.bib").read_text(encoding="utf-8")

    for key in [
        "qwen2024qwen25",
        "chen2025pitfalls",
        "ananthanarayanan2026physics",
        "wang2025cacheprune",
        "arditi2024refusal",
        "zhang2026anydepth",
    ]:
        assert f"{{{key}," in bib
