import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

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
