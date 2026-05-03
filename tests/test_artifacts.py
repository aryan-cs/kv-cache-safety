from pathlib import Path

from cache_safety_erasure.utils.io import append_jsonl, read_jsonl, write_json


def test_json_artifacts_roundtrip(tmp_path: Path) -> None:
    write_json(tmp_path / "environment.json", {"ok": True})
    append_jsonl(tmp_path / "generations.jsonl", [{"prompt_id": "p1", "text": "hello"}])
    assert read_jsonl(tmp_path / "generations.jsonl")[0]["prompt_id"] == "p1"


def test_cache_stats_sink_preserves_existing_rows_on_resume(tmp_path: Path) -> None:
    import sys

    import pandas as pd

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _CacheStatsSink

    path = tmp_path / "cache_stats.parquet"
    first = _CacheStatsSink(path, resume=False)
    first.write([{"prompt_id": "p1", "seed": 0, "policy": "none", "decode_step": 0}])
    first.close()

    second = _CacheStatsSink(path, resume=True)
    second.write([{"prompt_id": "p2", "seed": 0, "policy": "kv_int4_sim", "decode_step": 0}])
    second.close()

    df = pd.read_parquet(path)
    assert list(df["prompt_id"]) == ["p1", "p2"]


def test_latex_table_export_escapes_policy_names(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from export_paper_assets import write_latex_table

    table_path = tmp_path / "table.tex"
    write_latex_table(
        table_path,
        ["policy", "policy_level_ssei"],
        [{"policy": "sliding_window__budget64", "policy_level_ssei": 0.125}],
        caption="Caption with SSEI.",
        label="tab:test",
    )

    text = table_path.read_text(encoding="utf-8")
    assert "sliding\\_window\\_\\_budget64" in text
    assert "0.125" in text
    assert "\\label{tab:test}" in text
