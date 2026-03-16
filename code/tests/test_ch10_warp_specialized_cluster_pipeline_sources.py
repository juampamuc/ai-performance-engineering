from pathlib import Path


def test_warp_specialized_cluster_pipeline_sources_avoid_hot_loop_nvtx() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "ch10" / "baseline_warp_specialized_cluster_pipeline.cu",
        repo_root / "ch10" / "optimized_warp_specialized_cluster_pipeline.cu",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert 'NVTX_RANGE("warmup")' not in text


def test_optimized_warp_specialized_cluster_pipeline_uses_graph_replay() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "ch10" / "optimized_warp_specialized_cluster_pipeline.cu").read_text(
        encoding="utf-8"
    )

    assert "cudaGraphLaunch(graph_exec, stream)" in text
    assert "cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)" in text


def test_optimized_warp_specialized_cluster_pipeline_double_buffers_tiles() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "ch10" / "optimized_warp_specialized_cluster_pipeline.cu").read_text(
        encoding="utf-8"
    )

    assert "cuda::pipeline_shared_state<cuda::thread_scope_block, 2>" in text
    assert "float* A_tiles[2]" in text
    assert "float* B_tiles[2]" in text
    assert "enqueue_tile(next_stage, next_tile)" in text
    assert "const size_t shared_bytes = 5ull * TILE_ELEMS * sizeof(float);" in text
