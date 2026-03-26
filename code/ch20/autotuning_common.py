"""Shared model and setup constants for the Chapter 20 autotuning pair."""

from __future__ import annotations

import torch
import torch.nn as nn

AUTOTUNING_SETUP_PREWARM_ITERS = 10


class AutotuneModel(nn.Module):
    """Pointwise-heavy block that benefits from compiler fusion."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 0.01
        y = y * self.scale + self.bias
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        return y
