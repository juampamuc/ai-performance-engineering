from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from core.common.device_utils import (
    get_usable_cuda_or_cpu,
    require_cuda_device,
    resolve_requested_device,
)


def test_require_cuda_device_returns_cuda_when_available() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=True):
        assert require_cuda_device("CUDA required") == torch.device("cuda")


def test_require_cuda_device_honors_local_rank_env() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=True):
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            assert require_cuda_device("CUDA required", local_rank_env="LOCAL_RANK") == torch.device("cuda:3")


def test_require_cuda_device_raises_custom_error_when_cuda_missing() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA required for test"):
            require_cuda_device("CUDA required for test")


def test_resolve_requested_device_prefers_cuda_zero_when_available() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=True):
        assert resolve_requested_device(None) == torch.device("cuda:0")


def test_resolve_requested_device_returns_cpu_when_cuda_missing_and_no_arg() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=False):
        assert resolve_requested_device(None) == torch.device("cpu")


def test_get_usable_cuda_or_cpu_returns_cpu_when_cuda_missing() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=False):
        assert get_usable_cuda_or_cpu() == torch.device("cpu")


def test_get_usable_cuda_or_cpu_emits_warning_and_falls_back_when_probe_fails() -> None:
    warnings: list[str] = []
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=True):
        with patch("core.common.device_utils.torch.zeros", side_effect=RuntimeError("boom")):
            assert get_usable_cuda_or_cpu(warning_handler=warnings.append) == torch.device("cpu")

    assert warnings == ["CUDA unavailable or unsupported (boom); falling back to CPU."]
