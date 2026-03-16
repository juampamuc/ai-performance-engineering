from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from core.utils.dotenv import find_repo_root, load_repo_dotenv, reset_repo_dotenv_cache


@contextmanager
def _preserve_env(*keys: str):
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key in keys:
            original = snapshot[key]
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def test_load_repo_dotenv_prefers_local_override_and_export_syntax(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "BASE_ONLY=from_env\nSHARED=base\nexport EXPORTED=value\nEMPTY=\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text("SHARED=local\nLOCAL_ONLY=from_local\n", encoding="utf-8")

    with _preserve_env("BASE_ONLY", "SHARED", "EXPORTED", "LOCAL_ONLY", "EMPTY"):
        reset_repo_dotenv_cache(tmp_path)
        loaded = load_repo_dotenv(tmp_path)

        assert loaded == (tmp_path / ".env", tmp_path / ".env.local")
        assert os.environ["BASE_ONLY"] == "from_env"
        assert os.environ["SHARED"] == "local"
        assert os.environ["EXPORTED"] == "value"
        assert os.environ["LOCAL_ONLY"] == "from_local"
        assert os.environ["EMPTY"] == ""


def test_load_repo_dotenv_cache_requires_reset_for_reloads(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RELOAD_ME=first\n", encoding="utf-8")

    with _preserve_env("RELOAD_ME"):
        reset_repo_dotenv_cache(tmp_path)
        load_repo_dotenv(tmp_path)
        assert os.environ["RELOAD_ME"] == "first"

        env_file.write_text("RELOAD_ME=second\n", encoding="utf-8")
        os.environ.pop("RELOAD_ME", None)
        load_repo_dotenv(tmp_path)
        assert "RELOAD_ME" not in os.environ

        reset_repo_dotenv_cache(tmp_path)
        load_repo_dotenv(tmp_path)
        assert os.environ["RELOAD_ME"] == "second"


def test_find_repo_root_walks_up_to_pyproject(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "core" / "analysis"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    assert find_repo_root(nested / "module.py") == repo_root
