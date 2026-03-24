"""MoE Optimization Journey: From Naive to Production-Speed.

This lab takes you on a journey from a deliberately slow MoE implementation
to production-quality performance by applying optimization techniques from
the AI Performance Engineering book.

Each level builds on the previous shared model, so the story stays cumulative.

Levels:
    0. Naive - Sequential experts, Python loops (baseline)
    1. Batched - Parallel expert execution
    2. Fused - SiLU*up fusion
    3. MemEfficient - Buffer reuse on the shared model
    4. Grouped - Sort by expert and run grouped expert work
    5. BMM Fusion - Vectorized scatter plus a single BMM expert path
    6. CUDA Graphs - Capture and replay the fused Level 5 path
    7. Compiled - torch.compile on top of the graph-friendly model

Usage with bench CLI:
    # Run all levels
    python -m cli.aisp bench run --targets labs/moe_optimization_journey

    # Run specific level
    python -m cli.aisp bench run --targets labs/moe_optimization_journey/level0_naive

    # Compare two benchmark result JSON files
    python -m cli.aisp bench compare-runs \
        --baseline artifacts/runs/level0_naive/benchmark_test_results.json \
        --candidate artifacts/runs/level7_compiled/benchmark_test_results.json
"""

from dataclasses import dataclass


@dataclass
class MoEConfig:
    """MoE model configuration for optimization benchmarks."""

    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_layers: int = 4
    num_attention_heads: int = 16
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    batch_size: int = 4
    seq_len: int = 512
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    use_parallel_experts: bool = False
    use_torch_compile: bool = False
    use_fp8: bool = False
    use_triton_kernels: bool = False
    use_expert_parallelism: bool = False
    use_cuda_graphs: bool = False
    compile_mode: str = "reduce-overhead"

    @property
    def total_params(self) -> int:
        """Estimate total parameters."""
        embed = self.vocab_size * self.hidden_size
        expert_params = 3 * self.hidden_size * self.intermediate_size
        moe_params = self.num_experts * expert_params * self.num_layers
        router_params = self.hidden_size * self.num_experts * self.num_layers
        attn_params = 4 * self.hidden_size * self.hidden_size * self.num_layers
        lm_head = self.vocab_size * self.hidden_size
        return embed + moe_params + router_params + attn_params + lm_head

    @property
    def active_params(self) -> int:
        """Estimate active parameters per token."""
        expert_params = 3 * self.hidden_size * self.intermediate_size
        active_moe = self.num_experts_per_tok * expert_params * self.num_layers
        embed = self.vocab_size * self.hidden_size
        attn_params = 4 * self.hidden_size * self.hidden_size * self.num_layers
        router_params = self.hidden_size * self.num_experts * self.num_layers
        lm_head = self.vocab_size * self.hidden_size
        return embed + active_moe + router_params + attn_params + lm_head


CONFIGS = {
    "tiny": MoEConfig(
        hidden_size=256,
        intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        num_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
    ),
    "small": MoEConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_experts=4,
        num_experts_per_tok=2,
        num_layers=1,
        num_attention_heads=32,
        vocab_size=32000,
        seq_len=128,
    ),
    "medium": MoEConfig(
        hidden_size=1024,
        intermediate_size=2816,
        num_experts=8,
        num_experts_per_tok=2,
        num_layers=8,
        num_attention_heads=16,
        vocab_size=32000,
    ),
    "large": MoEConfig(
        hidden_size=2048,
        intermediate_size=5504,
        num_experts=16,
        num_experts_per_tok=4,
        num_layers=16,
        num_attention_heads=32,
        vocab_size=32000,
    ),
}


def get_config(name: str = "small", **overrides) -> MoEConfig:
    """Get a preset configuration with optional overrides."""
    if name not in CONFIGS:
        name = "small"
    config = MoEConfig(**CONFIGS[name].__dict__)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

__all__ = [
    "MoEConfig",
    "get_config",
    "CONFIGS",
]
