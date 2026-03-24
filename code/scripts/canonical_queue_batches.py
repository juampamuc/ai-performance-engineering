#!/usr/bin/env python3
"""Emit decision-complete benchmark target batches for canonical-host remediation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


CHAPTER_EXPECTATION_BATCH: Dict[str, List[str]] = {
    "ch01": [
        "ch01:gemm",
        "ch01:gemm_batched",
        "ch01:gemm_strided",
    ],
    "ch07": [
        "ch07:async_prefetch",
        "ch07:copy_scalar",
        "ch07:copy_scalar_vectorized",
        "ch07:copy_uncoalesced",
        "ch07:copy_uncoalesced_coalesced",
        "ch07:float4_vector",
        "ch07:hbm_copy",
        "ch07:hbm_peak",
        "ch07:lookup",
        "ch07:matmul",
        "ch07:matmul_tiled",
        "ch07:memory_access",
        "ch07:tma_bulk_tensor_2d",
        "ch07:transpose",
        "ch07:transpose_padded",
    ],
    "ch08": [
        "ch08:hbm_cuda_vectorized",
        "ch08:occupancy_tuning",
    ],
    "ch09": [
        "ch09:cublas_gemm_fp4_perchannel",
        "ch09:cublaslt_gemm",
        "ch09:cublaslt_gemm_fp16",
        "ch09:cublaslt_gemm_fp8",
        "ch09:cute_dsl_nvfp4_gemm",
        "ch09:cutlass_gemm",
        "ch09:cutlass_gemm_fp16",
        "ch09:cutlass_gemm_fp4",
        "ch09:cutlass_gemm_fp4_all_concepts",
        "ch09:cutlass_gemm_fp4_perchannel",
        "ch09:fused_l2norm",
        "ch09:micro_tiling_matmul",
    ],
    "ch10": [
        "ch10:atomic_reduction",
        "ch10:cluster_group",
        "ch10:cluster_group_no_dsmem",
        "ch10:cluster_group_single_cta",
        "ch10:cluster_multicast",
        "ch10:cooperative_persistent",
        "ch10:double_buffered_pipeline",
        "ch10:dsmem_reduction",
        "ch10:flash_attn_tma_micro_pipeline",
        "ch10:pipeline_3stage",
        "ch10:tma_2d_pipeline",
        "ch10:warp_spec_pingpong",
        "ch10:warp_specialized_cluster_pipeline",
        "ch10:warp_specialized_pipeline",
        "ch10:warp_specialized_pipeline_enhanced",
    ],
    "ch11": [
        "ch11:warp_specialized_multistream",
        "ch11:warp_specialized_two_pipelines_driver",
    ],
    "ch12": [
        "ch12:dynamic_parallelism_device",
        "ch12:dynamic_parallelism_host",
        "ch12:uneven_partition",
        "ch12:uneven_static",
    ],
    "ch19": [
        "ch19:fp4_hardware_kernel",
        "ch19:kv_prefetch_overlap",
    ],
}

CHAPTER_DRIFT_TRIAGE: List[str] = [
    "ch01:performance",
    "ch04:torchcomms",
    "ch06:attention_ilp",
    "ch08:thresholdtma",
    "ch09:cublaslt_gemm_fp4",
    "ch13:memory_profiling",
    "ch13:regional_compile",
    "ch15:guided_decoding",
    "ch20:integrated_kv_cache",
]

CAPABILITY_VALIDATION_BATCH: Dict[str, List[str]] = {
    "multi_gpu_ch04": [
        "ch04:no_overlap",
        "ch04:nvshmem_training_example",
        "ch04:nvshmem_training_patterns",
        "ch04:bandwidth_benchmark_suite",
        "ch04:symmetric_memory",
        "ch04:nvshmem_vs_nccl_benchmark",
        "ch04:nvshmem_pipeline_parallel",
    ],
    "hopper_only": [
        "ch09:cutlass_gemm_fp8",
    ],
    "grace_only": [
        "ch02:grace_coherent_memory",
    ],
}

LAB_FAMILY_BATCHES: Dict[str, List[str]] = {
    "labs_train_distributed": [
        "labs/train_distributed:ddp",
        "labs/train_distributed:ddp_compression",
        "labs/train_distributed:ddp_compression_int8",
        "labs/train_distributed:ddp_compression_powersgd",
        "labs/train_distributed:ddp_flash",
        "labs/train_distributed:fsdp",
        "labs/train_distributed:fsdp2",
        "labs/train_distributed:pipeline_1f1b",
        "labs/train_distributed:pipeline_dualpipe",
        "labs/train_distributed:pipeline_dualpipev",
        "labs/train_distributed:pipeline_gpipe",
        "labs/train_distributed:zero1",
        "labs/train_distributed:zero3",
    ],
    "labs_flashattention4": [
        "labs/flashattention4:best_available_attention",
        "labs/flashattention4:best_available_attention_alibi",
        "labs/flashattention4:best_available_attention_alibi_windowed",
        "labs/flashattention4:best_available_attention_causal",
        "labs/flashattention4:best_available_attention_softcap",
        "labs/flashattention4:best_available_attention_windowed",
        "labs/flashattention4:flashattention4",
        "labs/flashattention4:flashattention4_alibi_windowed",
        "labs/flashattention4:flashattention4_causal",
        "labs/flashattention4:flashattention4_dense",
        "labs/flashattention4:flashattention4_softcap",
        "labs/flashattention4:flashattention4_windowed",
    ],
    "labs_persistent_decode": [
        "labs/persistent_decode:native_tma_prefill_decode",
        "labs/persistent_decode:paged_kv_offload_prefetch",
        "labs/persistent_decode:persistent_decode",
        "labs/persistent_decode:persistent_decode_cuda",
        "labs/persistent_decode:persistent_decode_full_and_piecewise",
        "labs/persistent_decode:persistent_decode_graphs",
        "labs/persistent_decode:right_sized_decode",
    ],
    "labs_decode_optimization": [
        "labs/decode_optimization:decode",
        "labs/decode_optimization:decode_fp4",
        "labs/decode_optimization:decode_graph_full",
        "labs/decode_optimization:decode_hf_cache",
        "labs/decode_optimization:decode_pinned",
    ],
}


def _flatten(groups: Iterable[Iterable[str]]) -> List[str]:
    targets: List[str] = []
    for group in groups:
        targets.extend(group)
    return targets


def _payload() -> Dict[str, object]:
    return {
        "chapter_expectation_batch": CHAPTER_EXPECTATION_BATCH,
        "chapter_drift_triage": CHAPTER_DRIFT_TRIAGE,
        "capability_validation_batch": CAPABILITY_VALIDATION_BATCH,
        "lab_family_batches": LAB_FAMILY_BATCHES,
    }


def _all_batch_names() -> List[str]:
    return [
        "chapter_expectation_batch",
        "chapter_drift_triage",
        "capability_validation_batch",
        "lab_family_batches",
    ]


def _resolve_batch(name: str) -> List[str]:
    if name == "chapter_expectation_batch":
        return _flatten(CHAPTER_EXPECTATION_BATCH.values())
    if name == "chapter_drift_triage":
        return list(CHAPTER_DRIFT_TRIAGE)
    if name == "capability_validation_batch":
        return _flatten(CAPABILITY_VALIDATION_BATCH.values())
    if name == "lab_family_batches":
        return _flatten(LAB_FAMILY_BATCHES.values())
    raise KeyError(f"unknown batch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List batch names and counts.")
    parser.add_argument("--batch", choices=_all_batch_names(), help="Emit one named batch.")
    parser.add_argument(
        "--format",
        choices=("json", "lines", "shell"),
        default="json",
        help="Output format for --batch.",
    )
    args = parser.parse_args()

    if args.list:
        for name in _all_batch_names():
            print(f"{name}\t{len(_resolve_batch(name))}")
        return 0

    if not args.batch:
        print(json.dumps(_payload(), indent=2, sort_keys=True))
        return 0

    targets = _resolve_batch(args.batch)
    if args.format == "json":
        print(json.dumps({"batch": args.batch, "targets": targets}, indent=2))
        return 0
    if args.format == "lines":
        for target in targets:
            print(target)
        return 0
    print(
        "python -m core.benchmark.bench_commands run --profile none --validity-profile strict "
        + " ".join(f"-t {target}" for target in targets)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
