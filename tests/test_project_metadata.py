from __future__ import annotations

import tomllib
from pathlib import Path

from cache_safety_erasure.project import (
    PROJECT_NAME,
    PROJECT_REPOSITORY_GIT_URL,
    PROJECT_REPOSITORY_URL,
)
from cache_safety_erasure.utils.io import environment_snapshot


def test_project_repository_metadata_is_renamed_repo() -> None:
    assert PROJECT_NAME == "kv-cache-safety"
    assert PROJECT_REPOSITORY_URL == "https://github.com/aryan-cs/kv-cache-safety"
    assert PROJECT_REPOSITORY_GIT_URL == "https://github.com/aryan-cs/kv-cache-safety.git"


def test_pyproject_repository_metadata_matches_code_constant() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    urls = pyproject["project"]["urls"]
    assert urls["Homepage"] == PROJECT_REPOSITORY_URL
    assert urls["Repository"] == PROJECT_REPOSITORY_URL
    assert urls["Issues"] == f"{PROJECT_REPOSITORY_URL}/issues"


def test_environment_snapshot_records_repository_identity() -> None:
    snapshot = environment_snapshot()

    assert snapshot["project"]["name"] == PROJECT_NAME
    assert snapshot["project"]["repository_url"] == PROJECT_REPOSITORY_URL
    assert snapshot["project"]["repository_git_url"] == PROJECT_REPOSITORY_GIT_URL
