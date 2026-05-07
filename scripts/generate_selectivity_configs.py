from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.config import dump_yaml

CHAT_SAFETY_SMOKE_SUITES = [
    "system_leakage",
    "refusal_safety",
    "adversarial_refusal_safety",
    "benign_overrefusal",
    "instruction_following",
    "capability_smoke",
]

CHAT_SAFETY_POWERED_SUITES = [
    "system_leakage",
    "public_system_leakage",
    "public_refusal_safety",
    "adversarial_refusal_safety",
    "public_benign_overrefusal",
    "public_xstest_safe",
    "public_capability_arc",
]

BASE_MODEL_SMOKE_SUITES = [
    "base_alignment_contrast",
    "instruction_following",
    "capability_smoke",
]

BASE_MODEL_POWERED_SUITES = [
    "base_alignment_contrast",
    "instruction_following",
    "public_capability_arc",
]

CHAT_SAFETY_POLICIES = [
    {"name": "none"},
    {"name": "sliding_window", "budget": 128},
    {"name": "sink_recent", "budget": 128, "sink_tokens": 8},
    {"name": "random_matched", "budget": 128, "seed": 991},
    {
        "name": "policy_pinned",
        "budget": 128,
        "sink_tokens": 8,
        "protected_spans": ["system", "policy"],
    },
    {
        "name": "user_pinned",
        "budget": 128,
        "sink_tokens": 8,
        "protected_spans": ["user"],
    },
]

BASE_MODEL_POLICIES = [
    {"name": "none"},
    {"name": "sliding_window", "budget": 128},
    {"name": "sink_recent", "budget": 128, "sink_tokens": 8},
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-model selectivity experiment configs from the registered panel."
    )
    parser.add_argument("--panel", type=Path, default=Path("configs/models/selectivity_panel.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--stage", choices=["smoke", "powered"], default="smoke")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Panel key to generate. Repeatable. Defaults to all primary models.",
    )
    parser.add_argument("--include-fallbacks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = set(args.models or [])
    panel = load_yaml(args.panel)
    entries = list(panel.get("models", []))
    if args.include_fallbacks:
        entries.extend(panel.get("fallbacks", []))

    written: list[Path] = []
    for entry in entries:
        key = str(entry["key"])
        if selected and key not in selected:
            continue
        config = experiment_config_for_panel_entry(entry, args.stage)
        output_path = args.output_dir / f"{config['run']['name']}.yaml"
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"{output_path} exists; use --overwrite to replace it.")
        dump_yaml(config, output_path)
        written.append(output_path)

    if selected and len(written) != len(selected):
        written_keys = {path.stem.replace(f"selectivity_h200_{args.stage}_", "") for path in written}
        missing = sorted(selected - written_keys)
        raise SystemExit(f"Unknown or skipped panel keys: {', '.join(missing)}")
    for path in written:
        print(path)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to generate selectivity configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def experiment_config_for_panel_entry(entry: dict[str, Any], stage: str) -> dict[str, Any]:
    key = str(entry["key"])
    track = str(entry.get("track", "chat_safety"))
    model = deepcopy(entry["model"])
    model.update(
        {
            "family": entry.get("family"),
            "track": track,
            "role": entry.get("role"),
            "access": entry.get("access", "public"),
            "license": entry.get("license"),
        }
    )
    if entry.get("context_length") is not None:
        model["context_length"] = entry["context_length"]
    if entry.get("replaces") is not None:
        model["replaces"] = entry["replaces"]

    is_smoke = stage == "smoke"
    prompt_suites = (
        BASE_MODEL_SMOKE_SUITES
        if track == "base_model" and is_smoke
        else BASE_MODEL_POWERED_SUITES
        if track == "base_model"
        else CHAT_SAFETY_SMOKE_SUITES
        if is_smoke
        else CHAT_SAFETY_POWERED_SUITES
    )
    return {
        "run": {
            "name": f"selectivity_h200_{stage}_{key}",
            "output_dir": "results",
            "resume": True,
        },
        "model": model,
        "generation": {
            "max_new_tokens": 96 if is_smoke else 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "capture_attentions": False,
        },
        "prompt_suites": prompt_suites,
        "cache_policies": BASE_MODEL_POLICIES if track == "base_model" else CHAT_SAFETY_POLICIES,
        "seeds": [0],
        "limit_per_suite": 2 if is_smoke else None,
    }


if __name__ == "__main__":
    main()
