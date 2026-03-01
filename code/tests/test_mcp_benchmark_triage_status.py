from __future__ import annotations

import json
from pathlib import Path

import mcp.mcp_server as mcp_server


def test_benchmark_triage_counts_failed_status_classes(tmp_path: Path) -> None:
    data = {
        "timestamp": "2026-03-01T00:00:00",
        "results": [
            {
                "chapter": "ch_test",
                "benchmarks": [
                    {
                        "example": "ok_case",
                        "best_speedup": 1.20,
                        "baseline_time_ms": 10.0,
                        "status": "succeeded",
                        "optimizations": [],
                    },
                    {
                        "example": "verify_case",
                        "best_speedup": 1.10,
                        "baseline_time_ms": 11.0,
                        "status": "failed_verification",
                        "optimizations": [],
                    },
                    {
                        "example": "error_case",
                        "best_speedup": 0.95,
                        "baseline_time_ms": 12.0,
                        "status": "failed_error",
                        "optimizations": [],
                    },
                    {
                        "example": "skipped_case",
                        "best_speedup": 1.00,
                        "baseline_time_ms": 13.0,
                        "status": "skipped",
                        "optimizations": [],
                    },
                ],
            }
        ],
    }
    data_path = tmp_path / "benchmark_test_results.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")

    result = mcp_server.tool_benchmark_triage({"data_file": str(data_path)})
    summary = result["summary"]

    assert summary["total_benchmarks"] == 4
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["skipped"] == 1
    assert summary["failure_classes"]["failed_error"] == 1
    assert summary["failure_classes"]["failed_verification"] == 1
