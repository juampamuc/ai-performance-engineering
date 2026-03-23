"""baseline_cutlass.py - Baseline GEMM without CUTLASS optimization.

Demonstrates standard GEMM without CUTLASS library optimization.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineCutlassBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: GEMM without CUTLASS optimization (standard PyTorch matmul)."""
    
    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        # Match optimized matrix size for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )
        self._verification_payload = None
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        # Baseline: Standard PyTorch matmul (FP16) without CUTLASS optimization
        # Using FP16 to match optimized version for fair comparison
        # Disable TF32 to use standard GEMM kernels (not CUTLASS-optimized)
        # Match optimized version backend settings for fair comparison
        # Match optimized version dtype for fair comparison
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.zeros(self.m, self.n, device=self.device, dtype=torch.float16)

    def benchmark_fn(self) -> None:
        """Benchmark: Standard GEMM."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_cutlass", enable=enable_nvtx):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Benchmark not initialized")
            self.C = torch.matmul(self.A, self.B)
        if self.A is None or self.B is None or self.C is None:
            raise RuntimeError("Benchmark not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "A": self.A.detach(),
                "B": self.B.detach(),
            },
            output=self.C.detach(),
            batch_size=1,
            parameter_count=0,
            output_tolerance=(0.1, 2.0),
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.C = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            backend_policy="fp32_strict",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_gemm_metrics
        return compute_gemm_metrics(
            self.m, self.n, self.k,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp16",
            bytes_per_element=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineCutlassBenchmark()
