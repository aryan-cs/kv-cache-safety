from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.config import parse_experiment_config
from cache_safety_erasure.evals.io import load_prompt_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate environment and configs before H200 sweeps.")
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-no-cuda", action="store_true")
    parser.add_argument("--expected-gpu-substring", default="H200")
    parser.add_argument("--min-cuda-memory-gb", type=float, default=100.0)
    parser.add_argument("--skip-model-config-check", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    if not args.allow_dirty and _git_status_short():
        failures.append("git working tree is dirty")
    if shutil.which("uv") is None:
        failures.append("uv is not installed or not on PATH")

    _check_cuda(args, failures)
    for config_path in args.config:
        _check_config(config_path, args, failures)

    if failures:
        print("H200 PREFLIGHT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("H200 PREFLIGHT PASSED")


def _check_cuda(args: argparse.Namespace, failures: list[str]) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        failures.append("torch is not installed")
        return
    if not torch.cuda.is_available():
        if not args.allow_no_cuda:
            failures.append("CUDA is not available")
        return
    if torch.cuda.device_count() < 1:
        failures.append("CUDA reports zero devices")
        return
    matched_gpu = False
    min_bytes = int(args.min_cuda_memory_gb * (1024**3))
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        name = props.name
        if args.expected_gpu_substring.lower() in name.lower():
            matched_gpu = True
        if props.total_memory < min_bytes:
            failures.append(
                f"CUDA device {idx} `{name}` has {props.total_memory / (1024**3):.1f} GiB; "
                f"need >= {args.min_cuda_memory_gb:.1f} GiB"
            )
    if args.expected_gpu_substring and not matched_gpu:
        failures.append(f"no CUDA device name contains `{args.expected_gpu_substring}`")


def _check_config(config_path: Path, args: argparse.Namespace, failures: list[str]) -> None:
    try:
        config, _raw = parse_experiment_config(config_path)
    except Exception as exc:
        failures.append(f"{config_path}: config parse failed: {exc}")
        return
    if config.model.provider == "mock":
        failures.append(f"{config_path}: H200 config uses mock model")
    if config.model.model_id.startswith("sshleifer/tiny"):
        failures.append(f"{config_path}: H200 config uses tiny smoke model")
    if config.generation.capture_attentions and "attention" not in config.run.name:
        failures.append(
            f"{config_path}: capture_attentions=true outside a diagnostic attention run"
        )
    for suite in config.prompt_suites:
        try:
            prompts = load_prompt_suite(suite)
        except Exception as exc:
            failures.append(f"{config_path}: failed to load prompt suite `{suite}`: {exc}")
            continue
        if not prompts:
            failures.append(f"{config_path}: prompt suite `{suite}` is empty")
    if not args.skip_model_config_check:
        _check_hf_model_config(config_path, config.model.model_id, config.model.local_files_only, failures)


def _check_hf_model_config(
    config_path: Path, model_id: str, local_files_only: bool, failures: list[str]
) -> None:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ModuleNotFoundError:
        failures.append("transformers is not installed")
        return
    try:
        AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as exc:
        failures.append(f"{config_path}: cannot load model config/tokenizer for `{model_id}`: {exc}")


def _git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


if __name__ == "__main__":
    main()
