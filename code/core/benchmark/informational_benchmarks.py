from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Tuple

INFORMATIONAL_BENCHMARKS: Dict[str, Set[str]] = {
    # Ch4: DataParallel demo shows basic parallelism pattern (requires multi-GPU)
    "ch04": {"dataparallel_basic"},
    # Ch5: overlap/control demo remains useful, but no longer carries a canonical speed claim.
    "ch05": {"ai"},
    # Ch6: launch-bounds examples are small-effect teaching cases, not canonical speed claims.
    "ch06": {"launch_bounds", "launch_bounds_cuda"},
    # Ch8: custom-vs-library tcgen05 control pair is a narrative/control surface, not a headline speed claim.
    "ch08": {"tcgen05_custom_vs_cublas"},
    # Ch12: Graph CUDA demos show graph capture patterns.
    "ch12": {"graph_cuda", "cuda_graphs_conditional"},
    # Ch13: compound optimization, exploratory KV-cache, and torchao FP8 recipe demos stay noncanonical.
    "ch13": {
        "torchao_quantization_compiled",
        "kv_cache_naive_flash_blockwise",
        "precisionfp8",
        "precisionfp8_rowwise",
        "precisionfp8_rowwise_gw_hp",
    },
    # Ch14: explicit cuBLAS-vs-CUTLASS remains a supplementary control pair, not a chapter-native speed claim.
    "ch14": {"cublas_vs_cutlass"},
    # Ch15: Inference placement demo shows architecture patterns (multi-GPU).
    "ch15": {"inference_placement"},
    # Ch16: Hardware-variant dense-attention path and piece-graphs example remain informational.
    "ch16": {"dense_attention_flash_blackwell_variant", "piece_graphs"},
    # Ch17: Pipeline parallelism, routing demos, and the inference control pair remain informational.
    "ch17": {"pipeline_parallelism", "prefill_decode_disagg", "inference_full"},
    # Ch18: Speculative decoding demos show technique patterns.
    "ch18": {"speculative_decoding_multi_draft", "flexdecoding_graphs"},
    # Ch19: NVFP4 is new and may not be faster than BF16 yet.
    "ch19": {"nvfp4_training"},
    # Ch20: overlap demo remains informational until it is re-established as a stable speedup target.
    "ch20": {"pipeline_sequential"},
    # Labs: Dynamic router demos show routing patterns.
    "dynamic_router": {"dynamic_router", "router_vectorized"},
    # Labs: standalone pinned-host decode is a stepping-stone/control variant; the canonical
    # host-overhead claim stays on decode_streams where the benchmark actually carries a large
    # staged payload through the host-transfer path.
    "decode_optimization": {"decode_pinned"},
    # Labs: Full-stack tcgen05 follow-up is useful for profiling, but the canonical speed claim
    # remains on the coarse cluster_gemm path where the end-to-end win is material.
    "fullstack_cluster": {"cluster_gemm_tcgen05"},
    # Labs: the low-warp occupancy schedule is a useful Proton-vs-Nsight reference, but the
    # canonical speed claims stay on the main proton_matmul target and the larger winning tiles.
    "occupancy_tuning": {"proton_matmul_bm64_bn64_bk32_nw2"},
    # Labs: Persistent decode transport/control demos stay informational; canonical wins come from
    # the graph-backed decode path, TMA prefill, and the explicit paged-KV prefetch overlap pair.
    "persistent_decode": {
        "kv_locality_microbench",
        "persistent_decode_cuda",
        "nvlink_offload",
        "paged_kv_offload",
    },
    # Labs: grouped-GEMM case0-2 remain useful routing/control references, but on the current
    # virtualized B200 host they only show small deltas; keep canonical speed claims on case3
    # and the stricter ABAB/router tooling instead of sweep-gating these three surfaces.
    "nvfp4_group_gemm": {
        "nvfp4_group_gemm_case0",
        "nvfp4_group_gemm_case1",
        "nvfp4_group_gemm_case2",
    },
}


def informational_scope_candidates(scope: str) -> Tuple[str, ...]:
    normalized = str(scope).strip().replace("\\", "/")
    if not normalized:
        return ()

    candidates: list[str] = []

    def _add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    _add(normalized)
    underscored = normalized.replace("/", "_")
    _add(underscored)

    if normalized.startswith("labs/"):
        _add(Path(normalized).name)
        _add(normalized[len("labs/"):])
    if underscored.startswith("labs_"):
        _add(underscored[len("labs_"):])

    return tuple(candidates)


def is_informational_example(scope: str, example_name: str) -> bool:
    example = str(example_name).strip()
    if not example:
        return False
    return any(example in INFORMATIONAL_BENCHMARKS.get(candidate, set()) for candidate in informational_scope_candidates(scope))
