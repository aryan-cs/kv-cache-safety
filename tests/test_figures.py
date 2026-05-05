import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from check_publication_readiness import _check_figure_manifest, _figure_artifact_failure
from make_figures import (
    _paired_safety_forest_rows,
    _phase_portrait_rows,
    _prompt_effect_constellation_rows,
    _restoration_flow_rows,
    _safety_state_atlas_rows,
    _selective_rows_for_figures,
    _stream_cache_fingerprint,
    _stream_cache_summaries,
)


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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_stream_cache_fingerprint_uses_prompt_roles_and_position_bins(tmp_path: Path) -> None:
    import pandas as pd

    cache_path = tmp_path / "cache_stats.parquet"
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt_id":"p1","rendered_prompt":{"token_roles":["system","system","user","user"]}}\n',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "policy": "sliding_window__budget2",
                "decode_step": 0,
                "original_seq_len": 4,
                "retained_indices": "2,3",
                "evicted_indices": "0,1",
            },
            {
                "prompt_id": "p1",
                "policy": "sliding_window__budget2",
                "decode_step": 1,
                "original_seq_len": 4,
                "retained_indices": "0,1",
                "evicted_indices": "2,3",
            },
        ]
    ).to_parquet(cache_path, index=False)

    rows = _stream_cache_fingerprint(cache_path, prompts_path, bin_count=4)

    assert {
        "policy": "sliding_window__budget2",
        "layer_bin": 0,
        "layer_label": "L00",
        "layer_source": "unlayered_cache_rows",
        "layer_source_label": "unlayered cache rows",
        "role": "system",
        "token_bin": 0,
        "retained_count": 0.0,
        "evicted_count": 1.0,
        "retention_fraction": 0.0,
    } in rows
    assert {
        "policy": "sliding_window__budget2",
        "layer_bin": 0,
        "layer_label": "L00",
        "layer_source": "unlayered_cache_rows",
        "layer_source_label": "unlayered cache rows",
        "role": "user",
        "token_bin": 2,
        "retained_count": 1.0,
        "evicted_count": 0.0,
        "retention_fraction": 1.0,
    } in rows


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_stream_cache_fingerprint_synthesizes_layer_bands_from_legacy_rows(
    tmp_path: Path,
) -> None:
    import pandas as pd

    cache_path = tmp_path / "cache_stats.parquet"
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt_id":"p1","rendered_prompt":{"token_roles":["system","user"]}}\n',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "policy": "sliding_window__budget1",
                "seed": 0,
                "decode_step": 0,
                "original_seq_len": 2,
                "layer_count": 4,
                "retained_indices": "1",
                "evicted_indices": "0",
            }
            for _layer in range(4)
        ]
    ).to_parquet(cache_path, index=False)

    rows = _stream_cache_fingerprint(
        cache_path,
        prompts_path,
        bin_count=2,
        layer_bin_count=2,
    )

    observed_layers = {
        (row["layer_bin"], row["layer_label"]) for row in rows if row["role"] == "system"
    }
    assert observed_layers == {(0, "L00-L01"), (1, "L02-L03")}
    assert {row["layer_source"] for row in rows} == {"legacy_row_order"}
    assert {row["layer_source_label"] for row in rows} == {
        "legacy row-order layer inference"
    }


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_stream_cache_fingerprint_marks_explicit_layer_source(tmp_path: Path) -> None:
    import pandas as pd

    cache_path = tmp_path / "cache_stats.parquet"
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt_id":"p1","rendered_prompt":{"token_roles":["system","user"]}}\n',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "policy": "sliding_window__budget1",
                "decode_step": 0,
                "original_seq_len": 2,
                "layer": 3,
                "layer_count": 4,
                "retained_indices": "1",
                "evicted_indices": "0",
            }
        ]
    ).to_parquet(cache_path, index=False)

    rows = _stream_cache_fingerprint(cache_path, prompts_path, bin_count=2)

    assert {row["layer_label"] for row in rows} == {"L03"}
    assert {row["layer_source"] for row in rows} == {"explicit_layer"}
    assert {row["layer_source_label"] for row in rows} == {"explicit layer column"}


def test_phase_portrait_rows_parse_policy_budgets() -> None:
    import pandas as pd

    rows = _phase_portrait_rows(
        pd.DataFrame(
            [
                {
                    "suite": "public_refusal_safety",
                    "policy": "sliding_window__budget64",
                    "index": 0.2,
                    "safety_degradation": 0.3,
                    "capability_degradation": 0.1,
                }
            ]
        )
    )

    assert rows.to_dict(orient="records") == [
        {
            "suite": "public_refusal_safety",
            "policy": "sliding_window__budget64",
            "policy_family": "sliding_window",
            "budget_sort": 64.0,
            "budget_label": "b=64",
            "safety_degradation": 0.3,
            "capability_degradation": 0.1,
            "selective_safety_erasure_index": 0.2,
        }
    ]


def test_selective_rows_for_figures_falls_back_to_policy_level_contrasts() -> None:
    rows = _selective_rows_for_figures(
        {
            "selective_safety_erasure": {
                "public_refusal_safety::kv_int4_sim": {
                    "safety_degradation": 0.12,
                    "capability_degradation": None,
                    "selective_safety_erasure_index": None,
                }
            },
            "policy_level_contrasts": {
                "kv_int4_sim": {
                    "safety_degradation_ci": {
                        "mean": 0.12,
                        "ci_low": 0.08,
                        "ci_high": 0.16,
                        "n": 200,
                    },
                    "capability_degradation_ci": {
                        "mean": 0.02,
                        "ci_low": 0.0,
                        "ci_high": 0.04,
                        "n": 100,
                    },
                    "selective_safety_erasure_index": 0.10,
                    "selective_safety_erasure_index_ci": {
                        "ci_low": 0.04,
                        "ci_high": 0.15,
                        "n_safety": 200,
                        "n_capability": 100,
                    },
                }
            },
        }
    )

    assert rows == [
        {
            "suite_policy": "global_policy_contrast::kv_int4_sim",
            "suite": "global_policy_contrast",
            "policy": "kv_int4_sim",
            "contrast_scope": "policy_level",
            "index": 0.10,
            "selective_safety_erasure_index": 0.10,
            "safety_degradation": 0.12,
            "capability_degradation": 0.02,
            "safety_ci_low": 0.08,
            "safety_ci_high": 0.16,
            "capability_ci_low": 0.0,
            "capability_ci_high": 0.04,
            "selective_safety_erasure_index_ci_low": 0.04,
            "selective_safety_erasure_index_ci_high": 0.15,
            "safety_n": 200,
            "capability_n": 100,
        }
    ]


def test_selective_rows_for_figures_prefers_suite_level_ssei() -> None:
    rows = _selective_rows_for_figures(
        {
            "selective_safety_erasure": {
                "joint_suite::kv_int4_sim": {
                    "safety_degradation": 0.12,
                    "capability_degradation": 0.02,
                    "selective_safety_erasure_index": 0.10,
                }
            },
            "policy_level_contrasts": {
                "kv_int4_sim": {
                    "safety_degradation_ci": {"mean": 0.5},
                    "capability_degradation_ci": {"mean": 0.1},
                    "selective_safety_erasure_index": 0.4,
                }
            },
        }
    )

    assert rows == [
        {
            "suite_policy": "joint_suite::kv_int4_sim",
            "suite": "joint_suite",
            "policy": "kv_int4_sim",
            "contrast_scope": "suite",
            "index": 0.10,
            "selective_safety_erasure_index": 0.10,
            "safety_degradation": 0.12,
            "capability_degradation": 0.02,
        }
    ]


def test_paired_safety_forest_rows_keep_csv_labels_single_line() -> None:
    rows = _paired_safety_forest_rows(
        {
            "selective_safety_erasure": {
                "public_refusal_safety::sliding_window__budget64": {
                    "paired_safety_degradation_ci": {
                        "mean": 0.1,
                        "ci_low": 0.05,
                        "ci_high": 0.15,
                        "cluster_n": 10,
                    }
                }
            }
        }
    )

    assert rows == [
        {
            "label": "public_refusal_safety / sliding_window__budget64",
            "mean": 0.1,
            "ci_low": 0.05,
            "ci_high": 0.15,
            "cluster_n": 10,
        }
    ]
    assert "\n" not in rows[0]["label"]


def test_prompt_effect_constellation_rows_pair_against_baseline() -> None:
    import pandas as pd

    rows = _prompt_effect_constellation_rows(
        pd.DataFrame(
            [
                {
                    "suite": "public_refusal_safety",
                    "prompt_id": "p1",
                    "seed": 0,
                    "policy": "none",
                    "safety_score": 1.0,
                    "capability_score": None,
                    "refusal_expected_accuracy": 1.0,
                    "leakage_avoidance_score": None,
                    "generated_word_count": 10.0,
                },
                {
                    "suite": "public_refusal_safety",
                    "prompt_id": "p1",
                    "seed": 0,
                    "policy": "kv_int4_sim",
                    "safety_score": 0.25,
                    "capability_score": None,
                    "refusal_expected_accuracy": 0.0,
                    "leakage_avoidance_score": None,
                    "generated_word_count": 20.0,
                },
            ]
        )
    )

    assert len(rows) == 1
    assert rows[0]["safety_score_delta"] == 0.75
    assert rows[0]["refusal_expected_accuracy_delta"] == 1.0
    assert rows[0]["effect_magnitude"] == 1.0


def test_safety_state_atlas_combines_ssei_and_role_retention() -> None:
    rows = _safety_state_atlas_rows(
        [
            {
                "suite": "public_refusal_safety",
                "policy": "sliding_window__budget64",
                "index": 0.4,
                "safety_degradation": 0.5,
                "capability_degradation": 0.1,
            }
        ],
        [
            {
                "policy": "sliding_window__budget64",
                "role": "system",
                "retention_fraction": 0.25,
            },
            {
                "policy": "sliding_window__budget64",
                "role": "user",
                "retention_fraction": 0.75,
            },
        ],
    )

    assert rows == [
        {
            "suite": "public_refusal_safety",
            "policy": "sliding_window__budget64",
            "selective_safety_erasure_index": 0.4,
            "safety_degradation": 0.5,
            "capability_degradation": 0.1,
            "retention_scope": "policy_global",
            "system_retention_fraction": 0.25,
            "user_retention_fraction": 0.75,
            "template_retention_fraction": None,
            "generated_retention_fraction": None,
        }
    ]


def test_restoration_flow_rows_preserve_confidence_intervals() -> None:
    import pandas as pd

    rows = _restoration_flow_rows(
        pd.DataFrame(
            [
                {
                    "suite": "public_refusal_safety",
                    "policy": "patch_rolesystem",
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.6,
                    "safety_restoration_ci_low": 0.5,
                    "safety_restoration_ci_high": 0.75,
                }
            ]
        )
    )

    assert rows.to_dict(orient="records") == [
        {
            "suite": "public_refusal_safety",
            "policy": "patch_rolesystem",
            "compressed_policy": "kv_int4_sim",
            "safety_restoration_fraction": 0.6,
            "safety_restoration_ci_low": 0.5,
            "safety_restoration_ci_high": 0.75,
            "safety_restoration_ci_width": 0.25,
            "label": "public_refusal_safety / patch_system",
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
                    "data_row_count": 0,
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(figures_dir, results_dir, failures, require_causal_patch=False)

    assert any("stale png hash" in failure for failure in failures)


def test_figure_manifest_rejects_malformed_visual_artifacts(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    for suffix in ["png", "svg", "pdf", "csv"]:
        (figures_dir / f"figure.{suffix}").write_text("not a valid figure\n", encoding="utf-8")
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
                    "png_sha256": file_sha256(figures_dir / "figure.png"),
                    "svg": str(figures_dir / "figure.svg"),
                    "svg_sha256": file_sha256(figures_dir / "figure.svg"),
                    "pdf": str(figures_dir / "figure.pdf"),
                    "pdf_sha256": file_sha256(figures_dir / "figure.pdf"),
                    "data_csv": str(figures_dir / "figure.csv"),
                    "data_csv_sha256": file_sha256(figures_dir / "figure.csv"),
                    "data_row_count": 0,
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(figures_dir, results_dir, failures, require_causal_patch=False)

    assert "figure `figure` has invalid png: missing PNG signature" in failures
    assert "figure `figure` has invalid svg: missing SVG root" in failures
    assert "figure `figure` has invalid pdf: missing PDF signature" in failures


def test_figure_artifact_signature_validator_accepts_real_headers(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    svg = tmp_path / "figure.svg"
    csv = tmp_path / "figure.csv"
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fig.savefig(png)
    plt.close(fig)
    pdf.write_bytes(_test_pdf_bytes())
    svg.write_text('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    csv.write_text("column\nvalue\n", encoding="utf-8")

    assert _figure_artifact_failure("png", png) == ""
    assert _figure_artifact_failure("pdf", pdf) == ""
    assert _figure_artifact_failure("svg", svg) == ""
    assert _figure_artifact_failure("data_csv", csv) == ""


def test_figure_artifact_validator_rejects_blank_pdf_page(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    pdf = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=144)
    with pdf.open("wb") as f:
        writer.write(f)

    assert (
        _figure_artifact_failure("pdf", pdf)
        == "PDF page 1 has no rendered content stream"
    )


def test_figure_artifact_validator_rejects_all_white_pdf_content(tmp_path: Path) -> None:
    pdf = tmp_path / "white.pdf"
    pdf.write_bytes(
        _test_pdf_bytes(b"q 1 1 1 rg 1 1 1 RG 0 0 300 144 re f 0 0 m 1 1 l S Q")
    )

    assert _figure_artifact_failure("pdf", pdf) == "PDF page 1 appears visually blank"


def test_figure_manifest_requires_named_figures(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    for suffix in ["png", "svg", "pdf", "csv"]:
        (figures_dir / f"present.{suffix}").write_text(suffix, encoding="utf-8")
    write_json(
        figures_dir / "manifest.json",
        {
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
            },
            "figures": [
                {
                    "name": "present",
                    "png": str(figures_dir / "present.png"),
                    "png_sha256": file_sha256(figures_dir / "present.png"),
                    "svg": str(figures_dir / "present.svg"),
                    "svg_sha256": file_sha256(figures_dir / "present.svg"),
                    "pdf": str(figures_dir / "present.pdf"),
                    "pdf_sha256": file_sha256(figures_dir / "present.pdf"),
                    "data_csv": str(figures_dir / "present.csv"),
                    "data_csv_sha256": file_sha256(figures_dir / "present.csv"),
                    "data_row_count": 0,
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(
        figures_dir,
        results_dir,
        failures,
        require_causal_patch=False,
        required_figures=["missing_creative_figure"],
    )

    assert "missing required figure `missing_creative_figure`" in failures


def test_figure_manifest_rejects_blank_png(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    png = figures_dir / "figure.png"
    pdf = figures_dir / "figure.pdf"
    svg = figures_dir / "figure.svg"
    csv_path = figures_dir / "figure.csv"
    _write_blank_png(png)
    pdf.write_bytes(_test_pdf_bytes())
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>\n', encoding="utf-8")
    csv_path.write_text("x,y\n1,1\n", encoding="utf-8")
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
                    "png": str(png),
                    "png_sha256": file_sha256(png),
                    "svg": str(svg),
                    "svg_sha256": file_sha256(svg),
                    "pdf": str(pdf),
                    "pdf_sha256": file_sha256(pdf),
                    "data_csv": str(csv_path),
                    "data_csv_sha256": file_sha256(csv_path),
                    "data_row_count": 1,
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(figures_dir, results_dir, failures, require_causal_patch=False)

    assert "figure `figure` has invalid png: PNG appears visually blank" in failures


def _test_pdf_bytes(stream: bytes | None = None) -> bytes:
    if stream is None:
        stream = b"BT /F1 12 Tf 72 72 Td (Figure evidence) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(content)
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        content += f"{offset:010d} 00000 n \n".encode("ascii")
    content += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return content


def _write_blank_png(path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
