from __future__ import annotations

from pathlib import Path

from ch02.optimized_memory_transfer import OptimizedMemoryTransferBenchmark
from ch05.optimized_storage_cpu import OptimizedStorageCpuBenchmark
from core.discovery import discover_benchmarks

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ch02_optimized_memory_transfer_docstring_matches_async_pinned_copy() -> None:
    module_doc = (REPO_ROOT / "ch02" / "optimized_memory_transfer.py").read_text(encoding="utf-8")
    class_doc = OptimizedMemoryTransferBenchmark.__doc__

    assert "Pinned host memory with async H2D transfer" in module_doc
    assert class_doc is not None
    assert "asynchronous non-blocking copy" in class_doc
    assert "NVLink-C2C" not in module_doc
    assert "NVLink-C2C" not in class_doc


def test_ch04_no_overlap_docs_call_out_single_gpu_stand_in() -> None:
    ddp_no_overlap_text = (REPO_ROOT / "ch04" / "ddp_no_overlap.py").read_text(encoding="utf-8")
    wrapper_text = (REPO_ROOT / "ch04" / "baseline_no_overlap.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch04" / "README.md").read_text(encoding="utf-8")

    assert "Single-GPU simulation" in ddp_no_overlap_text
    assert "stand-in for" in ddp_no_overlap_text
    assert "all-reduce latency" in ddp_no_overlap_text
    assert "single-GPU simulation" in wrapper_text
    assert "stand-in for" in wrapper_text
    assert "Single-GPU overlap simulations" in readme_text


def test_ch05_optimized_storage_cpu_names_match_cpu_staged_path() -> None:
    source = (REPO_ROOT / "ch05" / "optimized_storage_cpu.py").read_text(encoding="utf-8")

    assert OptimizedStorageCpuBenchmark.__name__ == "OptimizedStorageCpuBenchmark"
    assert 'storage_cpu_optimized' in source
    assert "OptimizedStorageGdsBenchmark" not in source
    assert '"storage_gds"' not in source


def test_ch14_demo_files_stay_out_of_benchmark_discovery() -> None:
    ch14_dir = REPO_ROOT / "ch14"
    pairs = discover_benchmarks(ch14_dir)
    names = {example_name for _, _, example_name in pairs}

    assert (ch14_dir / "triton_persistent_demo.py").exists()
    assert (ch14_dir / "sliding_window_demo.py").exists()
    assert "triton_persistent" in names
    assert "sliding_window" in names
    assert "triton_persistent_demo" not in names
    assert "sliding_window_demo" not in names


def test_ch06_attention_ilp_docs_call_out_attention_score_microbenchmark() -> None:
    baseline_text = (REPO_ROOT / "ch06" / "baseline_attention_ilp.py").read_text(encoding="utf-8")
    optimized_text = (REPO_ROOT / "ch06" / "optimized_attention_ilp.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch06" / "README.md").read_text(encoding="utf-8")

    assert "attention-score" in baseline_text
    assert "attention-score" in optimized_text
    assert "multi-stream" not in optimized_text
    assert "scaled_dot_product_attention" not in optimized_text
    assert "attention-score preprocessing microbenchmark" in readme_text


def test_ch08_readme_marks_bridge_controls_vs_native_exemplars() -> None:
    readme_text = (REPO_ROOT / "ch08" / "README.md").read_text(encoding="utf-8")

    assert "chapter-native exemplars" in readme_text
    assert "bridge control" in readme_text
    assert "`threshold`, `loop_unrolling`, and `ai_optimization` are the chapter-native exemplars" in readme_text
    assert "`thresholdtma`, `tiling`, `tiling_tcgen05`, `tcgen05_custom_vs_cublas`, and `nvfp4_mlp` remain real baseline/optimized bridge controls" in readme_text
    assert "historical baseline/optimized filenames" not in readme_text


def test_ch17_inference_full_docs_mark_control_pair_not_disagg_exemplar() -> None:
    baseline_text = (REPO_ROOT / "ch17" / "baseline_inference_full.py").read_text(encoding="utf-8")
    optimized_text = (REPO_ROOT / "ch17" / "optimized_inference_full.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch17" / "README.md").read_text(encoding="utf-8")

    assert "control pair for model-side work reduction" in baseline_text
    assert "control pair for model-side work reduction" in optimized_text
    assert "disaggregated prefill/decode" in baseline_text
    assert "disaggregated prefill/decode" in optimized_text
    assert "It is not the" in baseline_text
    assert "It is not the" in optimized_text
    assert "chapter's native" in baseline_text
    assert "chapter's native" in optimized_text
    assert "inference_full` remains a control pair" in readme_text
    assert "chapter-native exemplars" in readme_text


def test_ch10_readme_stays_on_book_native_targets() -> None:
    readme_text = (REPO_ROOT / "ch10" / "README.md").read_text(encoding="utf-8")
    pairs = discover_benchmarks(REPO_ROOT / "ch10")
    names = {example_name for _, _, example_name in pairs}

    assert not (REPO_ROOT / "ch10" / "baseline_matmul.py").exists()
    assert not (REPO_ROOT / "ch10" / "optimized_matmul.py").exists()
    assert "baseline_matmul.py" not in readme_text
    assert "optimized_matmul.py" not in readme_text
    assert "double_buffered_pipeline" in readme_text
    assert "warp_specialized_pipeline" in readme_text
    assert "cluster_group" in readme_text
    assert "baseline_matmul_tcgen05_vs_cublas.py" in readme_text
    assert "baseline_tmem_tcgen05.py" not in readme_text
    assert "matmul" not in names
