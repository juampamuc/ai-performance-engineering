from pathlib import Path

from core.harness.run_benchmarks import (
    _format_failed_no_speedup,
    _should_fail_no_speedup,
    generate_markdown_report,
)


def test_should_fail_no_speedup_only_for_speed_goal_below_threshold() -> None:
    assert _should_fail_no_speedup({"optimization_goal": "speed", "best_speedup": 1.04}) is True
    assert _should_fail_no_speedup({"optimization_goal": "speed", "best_speedup": 1.05}) is False
    assert _should_fail_no_speedup({"optimization_goal": "memory", "best_speedup": 1.00}) is False


def test_generate_markdown_report_surfaces_failed_no_speedup(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    chapter_dir = bench_root / "ch01"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "baseline_example.py").write_text("def baseline():\n    return 1\n", encoding="utf-8")
    (chapter_dir / "optimized_example.py").write_text("def optimized():\n    return 1\n", encoding="utf-8")

    benchmark = {
        "example": "flat_speed",
        "type": "python",
        "baseline_file": "baseline_example.py",
        "baseline_time_ms": 10.0,
        "best_speedup": 1.02,
        "optimization_goal": "speed",
        "status": "failed_no_speedup",
        "error": _format_failed_no_speedup({"best_speedup": 1.02, "optimization_goal": "speed"}),
        "optimizations": [
            {
                "file": "optimized_example.py",
                "status": "succeeded",
                "time_ms": 9.8,
                "speedup": 1.02,
            }
        ],
    }
    results = [
        {
            "chapter": "ch01",
            "status": "completed",
            "benchmarks": [benchmark],
            "summary": {
                "total_benchmarks": 1,
                "successful": 0,
                "failed": 1,
                "failed_error": 0,
                "failed_verification": 0,
                "failed_regression": 0,
                "failed_no_speedup": 1,
                "failed_generic": 0,
                "failed_other": 0,
                "skipped_hardware": 0,
                "skipped_distributed": 0,
                "total_skipped": 0,
                "total_speedups": 0,
                "average_speedup": 1.02,
                "max_speedup": 1.02,
                "min_speedup": 1.02,
                "informational": 0,
            },
        }
    ]

    output_md = tmp_path / "benchmark_test_results.md"
    generate_markdown_report(results, output_md, bench_root=bench_root)

    report_text = output_md.read_text(encoding="utf-8")
    assert "No speedup (<1.05x, speed goal): 1" in report_text
    assert "No speedup: Best speedup 1.02x below required 1.05x threshold for speed-goal benchmark" in report_text
