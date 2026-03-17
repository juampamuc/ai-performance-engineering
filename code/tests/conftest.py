from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_benchmark_results_file(tmp_path: Path) -> Path:
    data = {
        "timestamp": "2025-01-01 00:00:00",
        "results": [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "example_a",
                        "best_speedup": 2.0,
                        "baseline_time_ms": 100.0,
                        "baseline_gpu_metrics": {"power_draw_w": 250},
                        "optimizations": [],
                        "status": "succeeded",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "benchmark_test_results.json"
    path.write_text(json.dumps(data))
    return path
