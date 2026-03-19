"""Blackwell/PTX support scaffold for the MoE CUDA/PTX lab."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from core.benchmark.blackwell_requirements import ensure_blackwell_tma_supported
from core.utils.extension_loader_template import load_cuda_extension_v2

_EXTENSION_DIR = Path(__file__).resolve().parent


def ensure_moe_ptx_supported() -> None:
    """Fail fast when the future PTX backend cannot run."""
    ensure_blackwell_tma_supported("labs/moe_cuda_ptx PTX backend")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError("SKIPPED: labs/moe_cuda_ptx PTX backend requires SM100-class GPUs.")
    if major == 12 and minor == 1:
        raise RuntimeError("SKIPPED: labs/moe_cuda_ptx PTX backend is not supported on sm_121.")


@lru_cache(maxsize=1)
def load_moe_ptx_extension():
    """Load the placeholder PTX extension used to keep milestone wiring honest."""
    ensure_moe_ptx_supported()
    return load_cuda_extension_v2(
        name="moe_cuda_ptx_stub_ext",
        sources=[_EXTENSION_DIR / "moe_cuda_ptx_stub.cu"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "-std=c++20"],
        extra_cflags=["-std=c++20"],
    )
