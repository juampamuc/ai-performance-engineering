from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import ch10.optimized_tcgen05_cluster_pipeline as optimized_tcgen05_cluster_pipeline
from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.harness.validity_checks import check_setup_precomputation


def test_optimized_tcgen05_cluster_pipeline_uses_cuda_graph_replay() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "ch10" / "optimized_tcgen05_cluster_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "torch.cuda.CUDAGraph()" in text
    assert "with torch.cuda.graph(self._graph, stream=self._graph_stream):" in text
    assert "self._graph.replay()" in text


def test_baseline_tcgen05_cluster_pipeline_remains_eager() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "ch10" / "baseline_tcgen05_cluster_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "CUDAGraph" not in text


def test_optimized_tcgen05_cluster_pipeline_setup_does_not_mutate_output_during_setup(monkeypatch) -> None:
    benchmark = object.__new__(optimized_tcgen05_cluster_pipeline.OptimizedTcgen05ClusterPipelineBenchmark)
    benchmark.extension = None
    benchmark._matmul = None
    benchmark._graph = None
    benchmark._graph_stream = None
    benchmark._output_buffer = None
    benchmark.output = None
    benchmark.device = "cuda"
    benchmark.matrix_a = None
    benchmark.matrix_b = None
    benchmark._synchronize = lambda: None

    monkeypatch.setattr(
        Tcgen05MatmulBenchmarkBase,
        "setup",
        lambda self: (
            setattr(self, "matrix_a", "matrix_a"),
            setattr(self, "matrix_b", "matrix_b"),
            setattr(self, "output", None),
        ),
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline,
        "ensure_dsmem_supported",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline,
        "ensure_tcgen05_supported",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline,
        "load_tcgen05_cluster_module",
        lambda: SimpleNamespace(matmul_tcgen05_cluster=lambda _a, _b: "captured-output"),
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline.torch,
        "inference_mode",
        lambda: nullcontext(),
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline.torch.cuda,
        "Stream",
        lambda device=None: object(),
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline.torch.cuda,
        "CUDAGraph",
        lambda: object(),
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline.torch.cuda,
        "stream",
        lambda _stream: nullcontext(),
    )
    monkeypatch.setattr(
        optimized_tcgen05_cluster_pipeline.torch.cuda,
        "graph",
        lambda _graph, stream=None: nullcontext(),
    )

    ok, error = check_setup_precomputation(
        lambda: {"output": getattr(benchmark, "output", None)},
        benchmark.setup,
    )

    assert ok is True
    assert error is None
    assert benchmark.output is None
    assert benchmark._output_buffer == "captured-output"
