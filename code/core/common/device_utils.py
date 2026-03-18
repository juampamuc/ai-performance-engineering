"""Lightweight device helpers shared across chapters."""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch


def resolve_local_rank(
    *,
    local_rank_env: str = "LOCAL_RANK",
    world_size_env: str = "WORLD_SIZE",
) -> int:
    """Resolve a local rank from the environment with multi-process validation."""
    raw_rank = os.environ.get(local_rank_env)
    if raw_rank not in (None, ""):
        try:
            return int(raw_rank)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid {local_rank_env} value {raw_rank!r}; expected an integer rank."
            ) from exc

    raw_world_size = os.environ.get(world_size_env)
    if raw_world_size in (None, ""):
        return 0

    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {world_size_env} value {raw_world_size!r}; expected an integer world size."
        ) from exc

    if world_size > 1:
        raise RuntimeError(f"{local_rank_env} must be set when {world_size_env} > 1")
    return 0


def get_preferred_device() -> Tuple[torch.device, Optional[str]]:
    """Return the best available device and an error message if CUDA is absent."""
    if torch.cuda.is_available():
        return torch.device("cuda"), None
    return torch.device("cpu"), "CUDA not available"


def cuda_supported() -> bool:
    """Convenience helper to check CUDA availability."""
    return torch.cuda.is_available()


def require_cuda_device(
    error_message: str,
    *,
    local_rank_env: Optional[str] = None,
) -> torch.device:
    """Return a CUDA device or raise a benchmark-specific error.

    When ``local_rank_env`` is set, resolve ``cuda:<rank>`` from the named
    environment variable so single-process distributed benchmarks honor the
    launcher-assigned local rank.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(error_message)
    if local_rank_env is None:
        return torch.device("cuda")
    rank = resolve_local_rank(local_rank_env=local_rank_env)
    return torch.device(f"cuda:{rank}")


def resolve_requested_device(device_arg: Optional[str]) -> torch.device:
    """Resolve an optional device argument with CUDA availability checks."""
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_usable_cuda_or_cpu(
    *,
    warning_handler: Optional[Callable[[str], None]] = None,
) -> torch.device:
    """Prefer CUDA but fall back to CPU when runtime probing fails."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
    except Exception as exc:
        message = f"CUDA unavailable or unsupported ({exc}); falling back to CPU."
        if warning_handler is not None:
            warning_handler(message)
        return torch.device("cpu")
    return torch.device("cuda")


__all__ = [
    "get_preferred_device",
    "cuda_supported",
    "resolve_local_rank",
    "require_cuda_device",
    "resolve_requested_device",
    "get_usable_cuda_or_cpu",
]
