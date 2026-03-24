from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import queue_restore


def test_queue_restore_prefers_queue_scripts_when_present() -> None:
    snapshot = {
        "queue_scripts": [
            {"cmd": "bash artifacts/parallel_runs/queue.sh"},
            {"cmd": "bash artifacts/parallel_runs/queue.sh"},
        ],
        "restore_commands": [
            "python -m cli.aisp bench run --targets ch01",
            "python -m cli.aisp bench run --targets ch02",
        ],
    }

    assert queue_restore._preferred_restore_commands(snapshot) == [
        "bash artifacts/parallel_runs/queue.sh"
    ]


def test_queue_restore_requires_command_index_for_multiple_commands() -> None:
    commands = [
        "python -m cli.aisp bench run --targets ch01",
        "python -m cli.aisp bench run --targets ch02",
    ]

    with pytest.raises(RuntimeError, match="--command-index"):
        queue_restore._select_restore_command(commands, None)


def test_queue_restore_spawn_command_supports_env_assignments(tmp_path: Path) -> None:
    output_path = tmp_path / "restored.txt"
    cmd = (
        f"RESTORE_TOKEN=ok python -c "
        f"\"import os; from pathlib import Path; "
        f"Path(r'{output_path}').write_text(os.environ['RESTORE_TOKEN'])\""
    )

    proc = queue_restore._spawn_command(cmd, str(tmp_path))
    assert proc.wait(timeout=10) == 0
    assert output_path.read_text(encoding="utf-8") == "ok"


def test_queue_restore_load_snapshot_validates_restore_commands(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps({"cwd": str(tmp_path), "restore_commands": ["python -m cli.aisp bench run --targets ch01"]}),
        encoding="utf-8",
    )

    loaded = queue_restore._load_snapshot(str(snapshot_path))
    assert loaded["restore_commands"] == ["python -m cli.aisp bench run --targets ch01"]
