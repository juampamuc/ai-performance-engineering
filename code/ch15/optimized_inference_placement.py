"""Optimized inference placement policy honoring NVLink-local TP/EP."""

from __future__ import annotations

from typing import Optional

from ch15.baseline_inference_placement import (  # noqa: E402
    _PlacementBenchmark,
    PlacementConfig,
)


class OptimizedInferencePlacementBenchmark(_PlacementBenchmark):
    """Heuristic-aligned placement: TP intra-node for prefill, TP=1 for decode, sticky sessions."""

    def __init__(self) -> None:
        cfg = PlacementConfig(
            prefill_tp_size=8,
            prefill_span_nodes=False,  # keep TP inside the NVLink island
            decode_tp_size=1,  # collapse TP for decode to kill all-reduce
            decode_span_nodes=False,
            decode_microbatch=4,
            remote_expert_fraction=0.05,  # expert pinning favors local shards
            router_sticky_decode=True,
            kv_transfer_policy="local_only",  # never walk KV across nodes mid-session
            notes="Prefill TP within node, decode TP=1, MoE local-first, KV stickiness.",
        )
        super().__init__(cfg, prefix="placement_optimized")


    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics for inference_placement."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=int(getattr(self, '_total_tokens', self.cfg.batch_size)),
            total_requests=int(getattr(self, '_total_requests', self.sessions)),
            batch_size=int(getattr(self, 'batch_size', self.cfg.batch_size)),
            max_batch_size=int(getattr(self, 'max_batch_size', self.cfg.batch_size)),
        )

def get_benchmark():
    return OptimizedInferencePlacementBenchmark()

