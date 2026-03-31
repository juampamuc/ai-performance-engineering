"""Tests for train_distributed benchmark-local contracts."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "labs" / "train_distributed"


def test_fsdp2_single_gpu_b200_contract_is_comparison() -> None:
    payload = json.loads((LAB_DIR / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = payload["examples"]["fsdp2"]

    assert entry["metadata"]["optimization_goal"] == "comparison"


def test_fsdp2_multi_gpu_b200_contract_stays_speed_gated() -> None:
    payload = json.loads((LAB_DIR / "expectations_2x_b200.json").read_text(encoding="utf-8"))
    entry = payload["examples"]["fsdp2"]

    assert entry["metadata"]["optimization_goal"] == "speed"
