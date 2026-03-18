from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from core.common.device_utils import (
    get_usable_cuda_or_cpu,
    require_cuda_device,
    resolve_local_rank,
    resolve_requested_device,
)


def test_require_cuda_device_returns_cuda_when_available() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=True):
        assert require_cuda_device("CUDA required") == torch.device("cuda")


def test_require_cuda_device_honors_local_rank_env() -> None:
    with patch("core.common.device_utils.torch.cuda.is_available", return_value=True):
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            assert require_cuda_device("CUDA required", local_rank_env="LOCAL_RANK") == torch.device("cuda:3")


def test_resolve_local_rank_defaults_to_zero_for_single_process() -> None:
    with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
        assert resolve_local_rank() == 0


def test_resolve_local_rank_uses_env_value_when_present() -> None:
    with patch.dict(os.environ, {"WORLD_SIZE": "8", "LOCAL_RANK": "5"}, clear=True):
        assert resolve_local_rank() == 5


def test_resolve_local_rank_supports_custom_env_names() -> None:
    with patch.dict(
        os.environ,
        {"OMPI_COMM_WORLD_SIZE": "4", "OMPI_COMM_WORLD_LOCAL_RANK": "2"},
        clear=True,
    ):
        assert (
            resolve_local_rank(
                local_rank_env="OMPI_COMM_WORLD_LOCAL_RANK",
                world_size_env="OMPI_COMM_WORLD_SIZE",
            )
            == 2
        )


def test_resolve_local_rank_requires_env_for_multi_process() -> None:
    with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True):
        with pytest.raises(RuntimeError, match="LOCAL_RANK must be set when WORLD_SIZE > 1"):
            resolve_local_rank()


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
