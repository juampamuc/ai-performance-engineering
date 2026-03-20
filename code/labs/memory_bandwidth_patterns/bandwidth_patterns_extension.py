"""CUDA extension loader for the memory-bandwidth-patterns lab."""

from __future__ import annotations

from pathlib import Path

import torch

from core.utils.extension_loader_template import load_cuda_extension_v2

_EXT_DIR = Path(__file__).resolve().parent


def _extension_name() -> str:
    if not torch.cuda.is_available():
        return "memory_bandwidth_patterns_kernels_cpu_only"
    major, minor = torch.cuda.get_device_capability()
    return f"memory_bandwidth_patterns_kernels_sm{major}{minor}"


def load_memory_bandwidth_patterns_extension():
    """Compile and load the local CUDA kernels for this lab."""

    if not torch.cuda.is_available():
        raise RuntimeError("labs.memory_bandwidth_patterns requires CUDA.")

    return load_cuda_extension_v2(
        name=_extension_name(),
        sources=[_EXT_DIR / "bandwidth_patterns_kernels.cu"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--use_fast_math",
            "-std=c++20",
        ],
        extra_cflags=["-O3", "-std=c++20"],
    )
