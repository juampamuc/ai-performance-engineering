"""optimized_inference_monolithic.py - Monolithic inference (optimized).

Pairs with: baseline_inference_monolithic.py

This variant keeps the same prefill+decode workload but reduces Python overhead
by running decode in a single call (instead of per-token calls) and avoids
explicit device-wide synchronizations inside the hot path.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata  # noqa: E402
from ch15.inference_monolithic_common import SimpleLLM
from ch15.verification_payload_mixin import VerificationPayloadMixin  # noqa: E402


class OptimizedInferenceMonolithicBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic inference optimized benchmark using shared harness conventions."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._decode_buffer: Optional[torch.Tensor] = None

        self.batch_size = 1
        self.prefill_seq = 64
        self.num_tokens = 128
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=self.prefill_seq + self.num_tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimpleLLM(vocab_size=10000, hidden_dim=512, num_layers=8).to(self.device).to(torch.bfloat16).eval()
        self.prompt = (torch.arange(self.prefill_seq, device=self.device, dtype=torch.int64) % 10000).unsqueeze(0)
        self.output = None
        self._decode_buffer = torch.empty(
            self.batch_size,
            self.num_tokens,
            self.model.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def benchmark_fn(self) -> None:
        if self.model is None or self.prompt is None or self._decode_buffer is None:
            raise RuntimeError("Model or prompt not initialized")

        with self._nvtx_range("inference_monolithic_optimized"):
            with torch.no_grad():
                kv_cache = self.model.prefill(self.prompt)
                self.output = self.model.decode_autoregressive(
                    kv_cache,
                    num_tokens=self.num_tokens,
                    output_buffer=self._decode_buffer,
                )

    def capture_verification_payload(self) -> None:
        if self.model is None or self.prompt is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"prompt": self.prompt},
            output=self.output.float(),
            batch_size=int(self.prompt.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.output = None
        self._decode_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "No output produced"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedInferenceMonolithicBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
