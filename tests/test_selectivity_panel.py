import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

import pytest
from generate_selectivity_configs import experiment_config_for_panel_entry, load_yaml
from merge_selectivity_panel_results import merge_runs
from preflight_h200 import _check_config
from score_base_model_track import _merge_base_metrics, compute_base_model_metrics

from cache_safety_erasure.config import parse_experiment_config


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_selectivity_panel_records_registered_primary_models() -> None:
    panel = load_yaml(Path("configs/models/selectivity_panel.yaml"))
    keys = {entry["key"] for entry in panel["models"]}

    assert {
        "gpt_oss_20b",
        "qwen2_5_7b_base",
        "qwen2_5_7b_instruct",
        "qwen3_5_9b",
        "llama3_1_8b_instruct",
        "gemma2_9b_it",
        "mistral_7b_instruct_v0_3",
        "olmo3_7b_instruct",
        "phi4",
    } <= keys
    assert any(entry["key"] == "phi4_mini_instruct" for entry in panel["fallbacks"])
    qwen3 = next(entry for entry in panel["models"] if entry["key"] == "qwen3_5_9b")
    assert qwen3["model"]["model_id"] == "Qwen/Qwen3-8B"
    assert "text_generation_replacement" in qwen3["role"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_generated_chat_safety_config_has_pinned_policy_controls() -> None:
    panel = load_yaml(Path("configs/models/selectivity_panel.yaml"))
    entry = next(entry for entry in panel["models"] if entry["key"] == "mistral_7b_instruct_v0_3")

    config = experiment_config_for_panel_entry(entry, "smoke")

    assert config["run"]["name"] == "selectivity_h200_smoke_mistral_7b_instruct_v0_3"
    assert config["model"]["family"] == "Mistral"
    assert config["model"]["track"] == "chat_safety"
    assert "system_leakage" in config["prompt_suites"]
    assert "adversarial_refusal_safety" in config["prompt_suites"]
    policy_names = {policy["name"] for policy in config["cache_policies"]}
    assert {"policy_pinned", "user_pinned", "random_matched"} <= policy_names
    user_pinned = next(policy for policy in config["cache_policies"] if policy["name"] == "user_pinned")
    assert user_pinned["protected_spans"] == ["user"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_generated_powered_config_uses_public_primary_suites() -> None:
    panel = load_yaml(Path("configs/models/selectivity_panel.yaml"))
    entry = next(entry for entry in panel["models"] if entry["key"] == "mistral_7b_instruct_v0_3")

    config = experiment_config_for_panel_entry(entry, "powered")

    assert "public_refusal_safety" in config["prompt_suites"]
    assert "public_system_leakage" in config["prompt_suites"]
    assert "adversarial_refusal_safety" in config["prompt_suites"]
    assert "refusal_safety" not in config["prompt_suites"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_gpt_oss_uses_compact_cache_position_mode() -> None:
    panel = load_yaml(Path("configs/models/selectivity_panel.yaml"))
    entry = next(entry for entry in panel["models"] if entry["key"] == "gpt_oss_20b")

    config = experiment_config_for_panel_entry(entry, "smoke")

    assert config["model"]["cache_position_mode"] == "compact"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_generated_base_model_config_excludes_chat_only_policies(tmp_path: Path) -> None:
    panel = load_yaml(Path("configs/models/selectivity_panel.yaml"))
    entry = next(entry for entry in panel["models"] if entry["key"] == "qwen2_5_7b_base")

    config = experiment_config_for_panel_entry(entry, "smoke")
    path = tmp_path / "base.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    parsed, _raw = parse_experiment_config(path)

    assert parsed.model.track == "base_model"
    assert parsed.model.chat_template_required is False
    assert "base_alignment_contrast" in parsed.prompt_suites
    assert "refusal_safety" not in parsed.prompt_suites
    assert {policy.name for policy in parsed.cache_policies} == {
        "none",
        "sliding_window",
        "sink_recent",
    }


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_h200_preflight_rejects_unpinned_panel_config(tmp_path: Path) -> None:
    panel = load_yaml(Path("configs/models/selectivity_panel.yaml"))
    entry = next(entry for entry in panel["models"] if entry["key"] == "qwen2_5_7b_instruct")
    config = experiment_config_for_panel_entry(entry, "smoke")
    config["model"]["revision"] = None
    path = tmp_path / "gemma.yaml"
    import argparse

    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    failures: list[str] = []

    _check_config(
        path,
        argparse.Namespace(skip_model_config_check=True),
        failures,
    )

    assert any("model revision must be pinned" in failure for failure in failures)


def test_base_model_metrics_report_safe_unsafe_margin_degradation() -> None:
    rows = [
        {
            "suite": "base_alignment_contrast",
            "prompt_id": "b1",
            "policy": "none",
            "safe_minus_unsafe_logprob_margin": 2.0,
            "unsafe_preferred": False,
        },
        {
            "suite": "base_alignment_contrast",
            "prompt_id": "b1",
            "policy": "sliding_window__budget128",
            "safe_minus_unsafe_logprob_margin": 0.5,
            "unsafe_preferred": False,
        },
    ]

    metrics = compute_base_model_metrics(rows)

    assert (
        metrics["base_alignment_contrasts"][
            "base_alignment_contrast::sliding_window__budget128"
        ]["margin_degradation"]
        == 1.5
    )


def test_base_model_metrics_merge_into_standard_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"policy_level_contrasts": {}}\n', encoding="utf-8")
    base_metrics = {"base_alignment_contrasts": {"base_alignment_contrast::sink_recent": {}}}

    _merge_base_metrics(tmp_path, base_metrics)

    merged = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert merged["base_model_metrics"] == base_metrics
    assert merged["base_alignment_contrasts"] == base_metrics["base_alignment_contrasts"]


def test_merge_selectivity_panel_preserves_family_rows(tmp_path: Path) -> None:
    run_dirs = []
    for family in ["Qwen", "Mistral"]:
        run_dir = tmp_path / family.lower()
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": f"{family}/model",
                    "model_family": family,
                    "model_track": "chat_safety",
                    "git_commit": "abc",
                    "git_dirty": False,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        rows = [
            {
                "suite": "public_refusal_safety",
                "prompt_id": "p1",
                "seed": 0,
                "policy": "none",
                "model_family": family,
                "model_track": "chat_safety",
                "safety_score": 1.0,
                "capability_score": None,
            },
            {
                "suite": "public_refusal_safety",
                "prompt_id": "p1",
                "seed": 0,
                "policy": "sliding_window__budget64",
                "model_family": family,
                "model_track": "chat_safety",
                "safety_score": 0.0,
                "capability_score": None,
            },
        ]
        (run_dir / "generations.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        run_dirs.append(run_dir)

    merged = merge_runs(run_dirs)

    assert merged["manifest"]["instruction_tuned_families"] == ["Mistral", "Qwen"]
    assert {row["source_run_id"] for row in merged["generations"]} == {"mistral", "qwen"}
