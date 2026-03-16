from __future__ import annotations

from pathlib import Path

from ch07.baseline_tma_copy import BaselineTMACopyBenchmark
from ch07.optimized_tma_copy import OptimizedTMACopyBenchmark


def test_tma_copy_wrappers_surface_the_neighbor_copy_story() -> None:
    baseline = BaselineTMACopyBenchmark()
    optimized = OptimizedTMACopyBenchmark()

    assert baseline.friendly_name == "Scalar Neighbor Gather Copy"
    assert optimized.friendly_name == "Pipeline + Tensor-Map Neighbor Copy"

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    assert baseline_metrics["copy.tensor_map_2d_requested"] == 0.0
    assert optimized_metrics["copy.tensor_map_2d_requested"] == 1.0


def test_ch07_readme_points_real_tma_story_to_bulk_tensor_target() -> None:
    readme = (Path(__file__).resolve().parents[1] / "ch07" / "README.md").read_text(encoding="utf-8")

    assert "`tma_bulk_tensor_2d`" in readme
    assert "neighbor-copy staging story" in readme
    assert "descriptor-backed TMA" in readme
