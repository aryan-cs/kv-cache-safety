from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read experiment configs. Run `uv sync --extra dev` first."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in config: {path}")
    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to write resolved configs.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: Path = Path("results")
    run_id: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    revision: str | None = None
    family: str | None = None
    track: str = "chat_safety"
    role: str | None = None
    access: str = "public"
    license: str | None = None
    context_length: int | None = None
    chat_template_required: bool = True
    cache_position_mode: str = "absolute"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    allow_cpu_offload: bool = False
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    local_files_only: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    stop_strings: tuple[str, ...] = field(default_factory=tuple)
    capture_attentions: bool = False


@dataclass(frozen=True)
class CachePolicyConfig:
    name: str
    budget: int | None = None
    sink_tokens: int = 4
    recent_tokens: int | None = None
    seed: int | None = None
    protected_spans: tuple[str, ...] = ("system", "policy")
    patch_from_baseline: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    generation: GenerationConfig
    prompt_suites: tuple[str, ...]
    cache_policies: tuple[CachePolicyConfig, ...]
    seeds: tuple[int, ...] = (0,)
    limit_per_suite: int | None = None


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def parse_experiment_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    config_path = Path(path)
    raw = _load_yaml(config_path)

    run_raw = raw.get("run", {})
    model_raw = raw.get("model", {})
    generation_raw = raw.get("generation", {})
    policies_raw = raw.get("cache_policies", [])

    if not model_raw:
        raise ValueError("Config must include a `model` section.")
    if not policies_raw:
        raise ValueError("Config must include at least one cache policy.")

    run = RunConfig(
        name=str(run_raw.get("name", config_path.stem)),
        output_dir=Path(run_raw.get("output_dir", "results")),
        run_id=run_raw.get("run_id"),
        resume=bool(run_raw.get("resume", False)),
    )
    model = ModelConfig(
        provider=str(model_raw.get("provider", "hf")),
        model_id=str(model_raw["model_id"]),
        revision=model_raw.get("revision"),
        family=model_raw.get("family"),
        track=str(model_raw.get("track", "chat_safety")),
        role=model_raw.get("role"),
        access=str(model_raw.get("access", "public")),
        license=model_raw.get("license"),
        context_length=None
        if model_raw.get("context_length") is None
        else int(model_raw["context_length"]),
        chat_template_required=bool(model_raw.get("chat_template_required", True)),
        cache_position_mode=str(model_raw.get("cache_position_mode", "absolute")),
        dtype=str(model_raw.get("dtype", "bfloat16")),
        device_map=str(model_raw.get("device_map", "auto")),
        allow_cpu_offload=bool(model_raw.get("allow_cpu_offload", False)),
        attn_implementation=model_raw.get("attn_implementation"),
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
        low_cpu_mem_usage=bool(model_raw.get("low_cpu_mem_usage", True)),
        local_files_only=bool(model_raw.get("local_files_only", False)),
    )
    generation = GenerationConfig(
        max_new_tokens=int(generation_raw.get("max_new_tokens", 128)),
        temperature=float(generation_raw.get("temperature", 0.0)),
        top_p=float(generation_raw.get("top_p", 1.0)),
        do_sample=bool(generation_raw.get("do_sample", False)),
        stop_strings=tuple(str(x) for x in _as_tuple(generation_raw.get("stop_strings"))),
        capture_attentions=bool(generation_raw.get("capture_attentions", False)),
    )

    policies = tuple(
        CachePolicyConfig(
            name=str(p["name"]),
            budget=None if p.get("budget") is None else int(p["budget"]),
            sink_tokens=int(p.get("sink_tokens", 4)),
            recent_tokens=None if p.get("recent_tokens") is None else int(p["recent_tokens"]),
            seed=None if p.get("seed") is None else int(p["seed"]),
            protected_spans=tuple(str(x) for x in _as_tuple(p.get("protected_spans", ("system", "policy")))),
            patch_from_baseline=p.get("patch_from_baseline"),
        )
        for p in policies_raw
    )

    prompt_suites = tuple(str(x) for x in _as_tuple(raw.get("prompt_suites")))
    if not prompt_suites:
        raise ValueError("Config must include at least one prompt suite.")

    experiment = ExperimentConfig(
        run=run,
        model=model,
        generation=generation,
        prompt_suites=prompt_suites,
        cache_policies=policies,
        seeds=tuple(int(x) for x in _as_tuple(raw.get("seeds", [0]))),
        limit_per_suite=None if raw.get("limit_per_suite") is None else int(raw["limit_per_suite"]),
    )
    return experiment, raw
