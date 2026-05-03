from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(output_dir: Path, name: str, run_id: str | None, resume: bool) -> Path:
    resolved_run_id = run_id or f"{name}_{utc_timestamp()}"
    run_dir = output_dir / resolved_run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Set run.resume=true to append safely."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_status_short": git_status_short(),
        "uv_lock_sha256": file_sha256(Path("uv.lock")),
        "cwd": str(Path.cwd()),
        "env": {
            key: os.environ.get(key)
            for key in [
                "HF_HOME",
                "HF_HUB_CACHE",
                "TRANSFORMERS_CACHE",
                "TORCH_HOME",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TOKENIZERS_PARALLELISM",
            ]
            if os.environ.get(key) is not None
        },
        "cgroup": cgroup_snapshot(),
        "packages": package_versions(
            ["torch", "transformers", "accelerate", "datasets", "numpy", "pandas", "pyarrow"]
        ),
    }
    try:
        import torch

        snapshot["torch_cuda_available"] = torch.cuda.is_available()
        snapshot["torch_mps_available"] = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if torch.cuda.is_available():
            snapshot["cuda_device_count"] = torch.cuda.device_count()
            snapshot["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except ModuleNotFoundError:
        snapshot["torch_available"] = False
    return snapshot


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cgroup_snapshot() -> dict[str, str | None]:
    paths = {
        "memory_max": Path("/sys/fs/cgroup/memory.max"),
        "memory_current": Path("/sys/fs/cgroup/memory.current"),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max"),
        "cpuset_cpus_effective": Path("/sys/fs/cgroup/cpuset.cpus.effective"),
    }
    snapshot: dict[str, str | None] = {}
    for key, path in paths.items():
        try:
            snapshot[key] = path.read_text(encoding="utf-8").strip()
        except OSError:
            snapshot[key] = None
    return snapshot


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    status = git_status_short()
    if status is None:
        return None
    return bool(status.strip())


def git_status_short() -> str | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required for cache_stats.parquet.") from exc
    pd.DataFrame(rows).to_parquet(path, index=False)
