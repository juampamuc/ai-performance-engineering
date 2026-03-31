"""Baseline serialized tiled loop for the software-pipelining lab."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.software_pipelining.software_pipelining_common import TilePipelineBenchmark


class BaselineTilePipelineBenchmark(TilePipelineBenchmark):
    def __init__(self) -> None:
        super().__init__(
            op_name="baseline_tile_pipeline",
            label="software_pipelining_baseline",
            pipeline_stage_count=1,
            notes="Serialized tile loop: load, then compute, with no next-tile overlap.",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineTilePipelineBenchmark()
