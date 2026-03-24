from __future__ import annotations

"""Strict distributed-launch helpers for Chapter 4 multi-GPU surfaces."""

import os
from typing import Callable, TypeVar

import torch

from core.common.device_utils import resolve_local_rank

_T = TypeVar("_T")


def setup_single_gpu_env(
    example_name: str = "distributed benchmark",
    *,
    min_world_size: int = 1,
    require_cuda: bool = True,
) -> tuple[int, int, int]:
    """Require a real distributed launch context instead of auto-seeding one.

    Historical Chapter 4 helpers silently fabricated `RANK/WORLD_SIZE=1`
    environments so multi-GPU demos would still "run" on unsupported hosts.
    That produced misleading results. Callers now fail fast with `SKIPPED:`
    diagnostics unless the required distributed context is already present.
    """

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"SKIPPED: {example_name} requires CUDA")

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(f"SKIPPED: {example_name} requires torchrun/distributed launch context")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = resolve_local_rank()

    if min_world_size > 1 and world_size < min_world_size:
        raise RuntimeError(f"SKIPPED: {example_name} requires >=2 GPUs")

    if require_cuda:
        available = torch.cuda.device_count()
        if available < min_world_size:
            raise RuntimeError(f"SKIPPED: {example_name} requires >=2 GPUs")
        if local_rank >= available:
            raise RuntimeError(
                f"SKIPPED: {example_name} requested local_rank={local_rank} but only {available} GPU(s) are visible"
            )

    return rank, world_size, local_rank


def run_main_with_skip_status(main_fn: Callable[[], _T]) -> int:
    """Execute a script entrypoint and convert capability skips into exit code 3."""

    try:
        main_fn()
    except RuntimeError as exc:
        message = str(exc).strip()
        if message.startswith("SKIPPED:"):
            print(message, flush=True)
            return 3
        raise
    return 0
