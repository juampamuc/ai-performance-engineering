from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Tuple

INFORMATIONAL_BENCHMARKS: Dict[str, Set[str]] = {
    # Ch4: DataParallel demo shows basic parallelism pattern (requires multi-GPU)
    "ch04": {"dataparallel_basic"},
    # Ch5: overlap/comparison demo remains useful, but no longer carries a canonical speed claim.
    "ch05": {"ai"},
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
    # Ch14: explicit cuBLAS-vs-CUTLASS remains a supplementary comparison pair, not a chapter-native speed claim.
    "ch14": {"cublas_vs_cutlass"},
    # Ch15: Inference placement demo shows architecture patterns (multi-GPU).
    "ch15": {"inference_placement"},
    # Ch16: Hardware-variant dense-attention path and piece-graphs example remain informational.
    "ch16": {"dense_attention_flash_blackwell_variant", "piece_graphs"},
    # Ch17: Pipeline parallelism, routing demos, and the inference comparison pair remain informational.
    "ch17": {"pipeline_parallelism", "prefill_decode_disagg", "inference_full"},
    # Ch18: Speculative decoding demos show technique patterns.
    "ch18": {"speculative_decoding_multi_draft", "flexdecoding_graphs"},
    # Ch19: NVFP4 is new and may not be faster than BF16 yet.
    "ch19": {"nvfp4_training"},
    # Ch20: overlap demo remains informational until it is re-established as a stable speedup target.
    "ch20": {"pipeline_sequential"},
    # Labs: Dynamic router demos show routing patterns.
    "dynamic_router": {"dynamic_router", "router_vectorized"},
    # Labs: Persistent decode transport/comparison demos stay informational; canonical wins come from
    # the graph-backed decode CUDA variant and KV-locality microbench.
    "persistent_decode": {
        "kv_locality_microbench",
        "persistent_decode_cuda",
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
