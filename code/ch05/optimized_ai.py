"""Optimized AI example: replay the tiny-block stack through a CUDA graph."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from ch05.ai_common import TinyBlock, compute_ai_workload_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedAIBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Chains the tiny blocks and replays them through a CUDA graph on GPU."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Sequential] = None
        self.static_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_output: Optional[torch.Tensor] = None
        self._graph_stream: Optional[torch.cuda.Stream] = None
        self._replay_fn: Optional[Callable[[], torch.Tensor]] = None
        # Must match baseline workload; tuned to keep the optimized path launch-bound
        # without making profiler runs pathological.
        self.batch = 64
        self.hidden = 32
        self.num_blocks = 256
        self.parameter_count = 0
        # Inference benchmark - jitter check not applicable
        tokens = self.batch * self.hidden * self.num_blocks
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        blocks = [TinyBlock(self.hidden) for _ in range(self.num_blocks)]
        self.model = nn.Sequential(*blocks).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.static_input = torch.randn(self.batch, self.hidden, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            _ = self.model(self.static_input)

        if self.device.type == "cuda":
            self._graph_stream = torch.cuda.Stream(device=self.device)
            self._graph = torch.cuda.CUDAGraph()

            self._synchronize()
            with torch.cuda.stream(self._graph_stream):
                with torch.inference_mode():
                    for _ in range(3):
                        _ = self.model(self.static_input)
                self._graph_stream.synchronize()
                with torch.inference_mode():
                    with torch.cuda.graph(self._graph, stream=self._graph_stream):
                        self._graph_output = self.model(self.static_input)
            self._synchronize()
            self._replay_fn = self._replay_graph
            return

        self._replay_fn = self._run_eager

    def _run_eager(self) -> torch.Tensor:
        if self.model is None or self.static_input is None:
            raise RuntimeError("Model/input not initialized")
        with torch.inference_mode():
            return self.model(self.static_input)

    def _replay_graph(self) -> torch.Tensor:
        if self._graph is None or self._graph_output is None:
            raise RuntimeError("CUDA graph not initialized")
        self._graph.replay()
        return self._graph_output

    def benchmark_fn(self) -> None:
        if self._replay_fn is None:
            raise RuntimeError("setup() must initialize replay path")
        with self._nvtx_range("optimized_ai"):
            out = self._replay_fn()
        self.output = out.detach()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.static_input},
            output=self.output,
            batch_size=self.batch,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.model = None
        self.static_input = None
        self.output = None
        self._graph = None
        self._graph_output = None
        self._graph_stream = None
        self._replay_fn = None
        self.parameter_count = 0
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return metrics derived from the actual model/workload shape."""
        return compute_ai_workload_metrics(
            batch_size=self.batch,
            hidden_dim=self.hidden,
            num_blocks=self.num_blocks,
            parameter_count=self.parameter_count,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.static_input is None:
            return "Model/input not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return OptimizedAIBenchmark()
