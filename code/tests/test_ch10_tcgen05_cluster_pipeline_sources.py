from pathlib import Path


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
