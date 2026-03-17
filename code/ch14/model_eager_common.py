"""Shared helpers for the Chapter 14 eager-vs-compile benchmark pair."""

from __future__ import annotations

import torch
import torch.nn as nn

MODEL_EAGER_WARMUP_ITERS = 20
MODEL_EAGER_COMPILE_WARMUP_ITERS = 50


def resolve_model_eager_dtype() -> torch.dtype:
    """Choose one reduced-precision dtype for both sides of the benchmark pair."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class SimpleTransformer(nn.Module):
    """Simple transformer used by both eager and compiled paths."""

    def __init__(self, d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=10000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, : x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
