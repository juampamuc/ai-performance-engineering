from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .matrix_types import MatrixPlaybook


PLAYBOOK_DIR = Path(__file__).with_name("playbooks")


def available_playbooks() -> list[str]:
    return sorted(path.stem for path in PLAYBOOK_DIR.glob("*.yaml"))


def resolve_playbook_path(spec: str | Path) -> Path:
    path = Path(spec)
    if path.exists():
        return path
    candidate = PLAYBOOK_DIR / f"{path.stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown playbook {spec!r}; available: {available_playbooks()}")


def load_playbook(spec: str | Path) -> MatrixPlaybook:
    path = resolve_playbook_path(spec)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Playbook {path} must decode to a mapping")
    data.setdefault("name", path.stem)
    data.setdefault("description", f"MoE decode matrix playbook loaded from {path.name}")
    return MatrixPlaybook.from_dict(data)
