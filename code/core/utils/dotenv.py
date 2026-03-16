"""Small shared loader for repo-local `.env` files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_DOTENV_CACHE: Dict[Tuple[str, bool, bool, bool], Tuple[Path, ...]] = {}


def find_repo_root(start: Path) -> Path:
    """Return the nearest repo root for a file or directory path."""
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current] + list(current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "core").is_dir():
            return candidate
    return current


def iter_repo_dotenv_files(repo_root: Path, *, include_local: bool = True) -> Iterable[Path]:
    """Yield repo-local dotenv files in load order."""
    resolved_root = repo_root.resolve()
    yield resolved_root / ".env"
    if include_local:
        yield resolved_root / ".env.local"


def _parse_env_line(raw_line: str) -> Tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if key.startswith("export "):
        key = key.replace("export", "", 1).strip()
    if not key:
        return None
    return key, value.strip().strip('"').strip("'")


def _load_env_file(env_file: Path, *, override: bool) -> bool:
    if not env_file.exists():
        return False

    with env_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
    return True


def load_repo_dotenv(
    repo_root: Path,
    *,
    include_local: bool = True,
    override: bool = False,
    local_override: bool = True,
    force: bool = False,
) -> Tuple[Path, ...]:
    """Load `.env` then `.env.local` from a repository root.

    `.env` only fills unset keys by default. `.env.local` overrides by default.
    Calls are cached per-root and option set unless `force=True`.
    """
    resolved_root = repo_root.resolve()
    cache_key = (str(resolved_root), include_local, override, local_override)
    if not force and cache_key in _DOTENV_CACHE:
        return _DOTENV_CACHE[cache_key]

    loaded: List[Path] = []
    for env_file in iter_repo_dotenv_files(resolved_root, include_local=include_local):
        applied = _load_env_file(
            env_file,
            override=override or (env_file.name == ".env.local" and local_override),
        )
        if applied:
            loaded.append(env_file)

    loaded_tuple = tuple(loaded)
    _DOTENV_CACHE[cache_key] = loaded_tuple
    return loaded_tuple


def reset_repo_dotenv_cache(repo_root: Path | None = None) -> None:
    """Clear cached dotenv loads globally or for a single repository root."""
    if repo_root is None:
        _DOTENV_CACHE.clear()
        return

    resolved_root = str(repo_root.resolve())
    stale_keys = [key for key in _DOTENV_CACHE if key[0] == resolved_root]
    for key in stale_keys:
        _DOTENV_CACHE.pop(key, None)
