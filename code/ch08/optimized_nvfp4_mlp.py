"""Optimized NVFP4 MLP for the Chapter 8 precision bridge-control pair."""

from __future__ import annotations

from core.benchmark.nvfp4_mlp import NVFP4MLPBenchmark, NVFP4MLPConfig
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedChapter8NVFP4MLPBenchmark(NVFP4MLPBenchmark):
    """Bridge control: NVFP4 path that points ahead to the tensor-core chapters."""

    def __init__(self) -> None:
        config = NVFP4MLPConfig(
            batch_size=512,
            d_model=8192,
            d_ff=32768,
            num_layers=2,
            iterations=20,
            warmup=10,
            name="ch08_nvfp4_mlp",
        )
        super().__init__(config, use_nvfp4=True)

    def get_custom_metrics(self) -> dict | None:
        return {
            "story.control_pair": 1.0,
            "story.chapter_native_exemplar": 0.0,
            "story.bridge_to_ch09": 1.0,
            "precision.nvfp4_enabled": 1.0,
            "model.d_model": float(self.config.d_model),
            "model.d_ff": float(self.config.d_ff),
            "model.layers": float(self.config.num_layers),
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedChapter8NVFP4MLPBenchmark()
