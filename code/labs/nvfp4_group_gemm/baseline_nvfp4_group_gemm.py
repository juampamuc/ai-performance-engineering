"""Canonical baseline NVFP4 grouped GEMM (promoted g2_n3072_k4096 shape).

This front-door target intentionally points at the currently promoted grouped-GEMM
shape so the public benchmark surface has one canonical speed target in addition to
the explicit shape-specific companions.
"""

from __future__ import annotations

import os

os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_UNROLL_N", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_ENABLE_EXPERIMENTAL_CTA2", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS_COMPRESS_LIST", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CAPTURE_ITER_GRAPH", "1")
os.environ.setdefault(
    "AISP_NVFP4_GROUP_GEMM_EXT_NAME",
    "nvfp4_group_gemm_tcgen05_baseline",
)

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.custom_cuda_submission import (
    custom_kernel_custom_cuda,
    prepare_custom_cuda,
)
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[2]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_custom_cuda,
        prepare=prepare_custom_cuda,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name="nvfp4_group_gemm_baseline_custom_cuda",
    )
    return attach_benchmark_metadata(bench, __file__)
