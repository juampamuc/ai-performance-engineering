"""Reference MXFP8-style quantization surface for the MoE CUDA/PTX lab."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda_ptx.moe_cuda_ptx_common import MoECudaPtxBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = MoECudaPtxBenchmark(
        target="moe_quant",
        backend="baseline",
        label="baseline_moe_quant",
    )
    return attach_benchmark_metadata(bench, __file__)
