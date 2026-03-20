from __future__ import annotations

import json
from pathlib import Path

from core.scripts import run_benchmark_pair_audit as audit_runner
from core.verification import review_baseline_optimized_pairs as pair_review


def test_report_drift_step_flags_status_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit_runner, "REPO_ROOT", tmp_path)

    chapter_dir = tmp_path / "ch11"
    chapter_dir.mkdir(parents=True)
    report_path = chapter_dir / audit_runner.REVIEW_REPORT_NAME
    report_path.write_text(
        "\n".join(
            [
                "# Chapter 11 Benchmark Pair Validity Review",
                "",
                "| Pair | Result | Notes |",
                "|---|---|---|",
                "| `demo` | FLAG | stale note |",
            ]
        ),
        encoding="utf-8",
    )

    baseline = chapter_dir / "baseline_demo.py"
    optimized = chapter_dir / "optimized_demo.py"
    baseline.write_text("class Dummy: pass\n", encoding="utf-8")
    optimized.write_text("class Dummy: pass\n", encoding="utf-8")
    pairs = {"ch11:demo": {"baseline": baseline, "optimized": optimized}}
    review_report = pair_review.ReviewReport(
        timestamp="2026-03-18T00:00:00+00:00",
        chapters=["ch11"],
        total_pairs=1,
        findings=[],
    )

    step, findings = audit_runner._run_report_drift_step(
        ["ch11"],
        pairs,
        review_report,
        validation_report=None,
        output_dir=tmp_path / "artifacts",
    )

    assert step["status"] == "completed_with_findings"
    assert len(findings) == 1
    assert findings[0]["issue_id"] == "PAIR_REVIEW_REPORT_DRIFT"


def test_build_scope_contracts_resolves_chapter_manuscript_and_fix_packet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_runner, "LABS_INDEX_PATH", tmp_path / "labs" / "README.md")

    book_after = tmp_path / "book-after"
    book_after.mkdir(parents=True)
    (book_after / "ch10.md").write_text("# Chapter 10\n", encoding="utf-8")
    (book_after / "ch10-fix.md").write_text("# Chapter 10 Fixes\n", encoding="utf-8")

    baseline = tmp_path / "ch10" / "baseline_demo.py"
    optimized = tmp_path / "ch10" / "optimized_demo.py"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("class Dummy: pass\n", encoding="utf-8")
    optimized.write_text("class Dummy: pass\n", encoding="utf-8")

    contracts, findings = audit_runner._build_scope_contracts(
        ["ch10"],
        {"ch10:demo": {"baseline": baseline, "optimized": optimized}},
    )

    assert findings == []
    assert contracts[0]["source_doc"].endswith("book-after/ch10.md")
    assert contracts[0]["supplemental_doc"].endswith("book-after/ch10-fix.md")
    assert contracts[0]["status"] == "PASS"


def test_build_scope_contracts_flags_benchmark_pair_lab_without_pairs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_runner, "LABS_INDEX_PATH", tmp_path / "labs" / "README.md")

    labs_dir = tmp_path / "labs"
    labs_dir.mkdir(parents=True)
    (labs_dir / "README.md").write_text(
        "\n".join(
            [
                "# Labs",
                "",
                "| Path | Description |",
                "|---|---|",
                "| `labs/block_scaling` | Benchmark-pair labs with strong kernel/perf narratives and artifact-backed measured deltas. |",
            ]
        ),
        encoding="utf-8",
    )
    block_scaling = labs_dir / "block_scaling"
    block_scaling.mkdir()
    (block_scaling / "README.md").write_text("# Block Scaling\n", encoding="utf-8")

    contracts, findings = audit_runner._build_scope_contracts(["labs/block_scaling"], {})

    assert contracts[0]["review_mode"] == "benchmark-pair"
    assert contracts[0]["status"] == "FLAG"
    assert any(finding["issue_id"] == "PAIR_SCOPE_CONTRACT_MISMATCH" for finding in findings)


def test_load_lab_classifications_includes_cache_aware_disagg_inference() -> None:
    classifications = audit_runner._load_lab_classifications()

    assert classifications["labs/cache_aware_disagg_inference"] == "benchmark-story"


def test_audit_main_writes_manifest_and_summary(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline_demo.py"
    optimized = tmp_path / "optimized_demo.py"
    baseline.write_text("class Dummy: pass\n", encoding="utf-8")
    optimized.write_text("class Dummy: pass\n", encoding="utf-8")
    pairs = {"ch11:demo": {"baseline": baseline, "optimized": optimized}}
    review_report = pair_review.ReviewReport(
        timestamp="2026-03-18T00:00:00+00:00",
        chapters=["ch11"],
        total_pairs=1,
        findings=[],
    )

    monkeypatch.setattr(audit_runner, "_discover_pairs", lambda scopes: pairs)
    monkeypatch.setattr(
        audit_runner,
        "_run_review_step",
        lambda scopes, output_dir: ({"status": "completed", "summary": {}, "artifacts": {}}, review_report),
    )
    monkeypatch.setattr(
        audit_runner,
        "_run_compliance_step",
        lambda files, output_dir: ({"status": "completed", "summary": {}, "artifacts": {}}, object()),
    )
    monkeypatch.setattr(
        audit_runner,
        "_run_pair_validation_step",
        lambda scopes, output_dir, cuda_available: ({"status": "skipped", "summary": {}, "artifacts": {}}, None),
    )
    monkeypatch.setattr(
        audit_runner,
        "_run_pytest_audit_step",
        lambda output_dir: {"status": "completed_with_findings", "summary": {}, "artifacts": {}},
    )
    monkeypatch.setattr(
        audit_runner,
        "_run_scope_contract_step",
        lambda scopes, pairs, output_dir: (
            {"status": "completed", "summary": {"scopes": 1, "findings": 0}, "artifacts": {}},
            [{"scope": "ch11", "scope_type": "chapter", "review_mode": "chapter-manuscript", "pair_count": 1, "source_doc": "book-after/ch11.md", "source_doc_exists": True, "supplemental_doc": None, "status": "PASS", "notes": []}],
            [],
        ),
    )
    monkeypatch.setattr(
        audit_runner,
        "_run_report_drift_step",
        lambda scopes, pairs, review_report, validation_report, output_dir: (
            {"status": "completed", "summary": {"findings": 0}, "artifacts": {}},
            [],
        ),
    )
    monkeypatch.setattr(
        audit_runner,
        "_run_gpu_rescan_step",
        lambda scopes, output_dir, include_gpu_rescan: {"status": "not_requested", "summary": {}, "artifacts": {}},
    )
    monkeypatch.setattr(
        audit_runner,
        "_pair_results",
        lambda pairs, review_report, validation_report, cuda_available: [
            {"pair_key": "ch11:demo", "scope": "ch11", "status": "PASS", "bucket": "static-only", "skip_reason": None}
        ],
    )
    monkeypatch.setattr(audit_runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(audit_runner.torch.cuda, "device_count", lambda: 0)

    output_dir = tmp_path / "audit"
    rc = audit_runner.main(["--scope", "ch11", "--output-dir", str(output_dir)])

    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert manifest["pair_count"] == 1
    assert summary["steps"]["pytest_audit"]["status"] == "completed_with_findings"
    assert summary["steps"]["scope_contract"]["status"] == "completed"
    assert (output_dir / "summary.md").exists()
