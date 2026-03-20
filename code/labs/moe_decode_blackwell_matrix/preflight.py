from __future__ import annotations

import shutil
from typing import Any

import torch


def require_blackwell_cuda(*, allow_non_blackwell: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This lab requires CUDA")
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    device_meta = {
        "device_index": device_index,
        "device_name": device_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if capability[0] < 10 and not allow_non_blackwell:
        raise RuntimeError(
            f"Expected a Blackwell-class GPU (sm_100+); got {device_name} "
            f"with compute capability {capability[0]}.{capability[1]}"
        )
    return device_meta


def query_app_clock_state(device_index: int = 0) -> dict[str, Any]:
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("pynvml is required to record locked application clocks") from exc

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return {
            "app_clock_sm_mhz": int(
                pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
            ),
            "app_clock_mem_mhz": int(
                pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)
            ),
            "cur_clock_sm_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            ),
            "cur_clock_mem_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            ),
        }
    finally:
        pynvml.nvmlShutdown()


def profiler_tool_status() -> dict[str, Any]:
    return {
        "torch_profiler_available": hasattr(torch, "profiler"),
        "nsys_path": shutil.which("nsys"),
        "ncu_path": shutil.which("ncu"),
    }
