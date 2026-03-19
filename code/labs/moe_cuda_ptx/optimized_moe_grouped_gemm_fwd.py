"""CUDA grouped expert forward FFN surface for the MoE CUDA/PTX lab."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda_ptx.moe_cuda_ptx_common import MoECudaPtxBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = MoECudaPtxBenchmark(
        target="moe_grouped_gemm_fwd",
        backend="cuda",
        label="optimized_moe_grouped_gemm_fwd",
    )
    return attach_benchmark_metadata(bench, __file__)
