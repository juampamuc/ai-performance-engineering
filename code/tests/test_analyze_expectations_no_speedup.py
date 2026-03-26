from pathlib import Path

from core.analysis.analyze_expectations import classify_entry


def test_classify_entry_treats_failed_no_speedup_as_fail(tmp_path: Path) -> None:
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "baseline_example.py").write_text("print('baseline')\n", encoding="utf-8")
    (chapter_dir / "optimized_example.py").write_text("print('ok')\n", encoding="utf-8")

    benchmark = {
        "example": "flat_speed",
        "status": "failed_no_speedup",
        "best_speedup": 1.02,
        "optimization_goal": "speed",
        "baseline_file": "baseline_example.py",
        "optimizations": [{"file": "optimized_example.py", "status": "succeeded", "speedup": 1.02}],
        "error": "Best speedup 1.02x below required 1.05x threshold for speed-goal benchmark",
    }

    entry = classify_entry(
        chapter_name="ch01",
        benchmark=benchmark,
        artifact_dir="runs/test",
        threshold=1.05,
        repo_root=repo_root,
    )

    assert entry is not None
    assert entry.severity == "fail"
    assert entry.optimization_goal == "speed"
