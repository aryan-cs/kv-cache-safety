import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from check_publication_readiness import _check_figure_manifest
from make_figures import _stream_cache_summaries


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_stream_cache_summaries_aggregates_without_full_read(tmp_path: Path) -> None:
    import pandas as pd

    cache_path = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "sliding_window__budget64",
                "decode_step": 0,
                "cache_l2_before": 4.0,
                "cache_l2_after": 2.0,
                "retained_system_tokens": 2,
                "evicted_system_tokens": 2,
            },
            {
                "policy": "sliding_window__budget64",
                "decode_step": 0,
                "cache_l2_before": 2.0,
                "cache_l2_after": 1.0,
                "retained_system_tokens": 1,
                "evicted_system_tokens": 3,
            },
        ]
    ).to_parquet(cache_path, index=False)

    summaries = _stream_cache_summaries(cache_path)

    assert summaries["l2_rows"] == [
        {
            "policy": "sliding_window__budget64",
            "decode_step": 0,
            "l2_retained_fraction": 0.5,
        }
    ]
    assert summaries["role_rows"] == [
        {
            "policy": "sliding_window__budget64",
            "role": "system",
            "retention_fraction": 0.375,
            "retained_count": 3.0,
            "evicted_count": 5.0,
        }
    ]


def test_figure_manifest_rejects_stale_hash(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    for suffix in ["png", "svg", "pdf", "csv"]:
        (figures_dir / f"figure.{suffix}").write_text(suffix, encoding="utf-8")
    write_json(
        figures_dir / "manifest.json",
        {
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
            },
            "figures": [
                {
                    "name": "figure",
                    "png": str(figures_dir / "figure.png"),
                    "png_sha256": "stale",
                    "svg": str(figures_dir / "figure.svg"),
                    "svg_sha256": file_sha256(figures_dir / "figure.svg"),
                    "pdf": str(figures_dir / "figure.pdf"),
                    "pdf_sha256": file_sha256(figures_dir / "figure.pdf"),
                    "data_csv": str(figures_dir / "figure.csv"),
                    "data_csv_sha256": file_sha256(figures_dir / "figure.csv"),
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(figures_dir, results_dir, failures, require_causal_patch=False)

    assert any("stale png hash" in failure for failure in failures)
