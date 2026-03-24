#!/usr/bin/env python3
"""Restore queued bench/profiling runs from a snapshot JSON."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from typing import Dict, List, Tuple


def _load_snapshot(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "restore_commands" not in data:
        raise KeyError("Snapshot missing restore_commands")
    if not isinstance(data["restore_commands"], list):
        raise TypeError("restore_commands must be a list")
    return data


def _spawn_command(cmd: str, cwd: str) -> subprocess.Popen:
    tokens = shlex.split(cmd)
    if not tokens:
        raise ValueError("Restore command is empty")

    env_updates: Dict[str, str] = {}
    while tokens and "=" in tokens[0]:
        key, value = tokens[0].split("=", 1)
        if not key.isidentifier():
            break
        env_updates[key] = value
        tokens.pop(0)

    if not tokens:
        raise ValueError("Restore command only contained environment assignments")

    env = os.environ.copy()
    env.update(env_updates)
    return subprocess.Popen(
        tokens,
        cwd=cwd,
        env=env,
    )


def _preferred_restore_commands(snapshot: dict) -> List[str]:
    queue_scripts = snapshot.get("queue_scripts") or []
    queue_cmds = []
    for record in queue_scripts:
        if isinstance(record, dict):
            cmd = record.get("cmd")
            if isinstance(cmd, str) and cmd not in queue_cmds:
                queue_cmds.append(cmd)
    if queue_cmds:
        return queue_cmds
    return list(snapshot["restore_commands"])


def _select_restore_command(commands: List[str], command_index: int | None) -> Tuple[str, bool]:
    if command_index is not None:
        if command_index < 0 or command_index >= len(commands):
            raise IndexError(f"command-index {command_index} out of range for {len(commands)} restore command(s)")
        return commands[command_index], True

    if len(commands) != 1:
        choices = "\n".join(f"  [{idx}] {cmd}" for idx, cmd in enumerate(commands))
        raise RuntimeError(
            "Snapshot contains multiple restore commands; rerun with --command-index to restore exactly one.\n"
            f"Available commands:\n{choices}"
        )
    return commands[0], False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to snapshot JSON produced by scripts/queue_snapshot.py.",
    )
    parser.add_argument(
        "--command-index",
        type=int,
        default=None,
        help="Restore exactly one command from the snapshot by index.",
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args.snapshot)
    restore_commands = _preferred_restore_commands(snapshot)
    if not restore_commands:
        raise RuntimeError("No restore commands in snapshot")

    cwd = snapshot.get("cwd") or os.getcwd()
    cmd, selected_explicitly = _select_restore_command(restore_commands, args.command_index)
    if len(restore_commands) > 1 and selected_explicitly:
        print(f"[queue_restore] selected command[{args.command_index}] from {len(restore_commands)} candidates")
    print(f"[queue_restore] launching: {cmd}")
    proc = _spawn_command(cmd, cwd)
    print(f"[queue_restore] spawned pid={proc.pid} (cwd={cwd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
