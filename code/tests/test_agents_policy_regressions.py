from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_agents_test_realism_policy_distinguishes_benchmark_truth_from_orchestration() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "Benchmark-truth tests MUST execute real repo code paths end-to-end." in agents
    assert "Narrow control-plane/orchestration tests MAY use `monkeypatch`/`patch`" in agents
    assert "Tests MUST NOT use mocks, stubs, `monkeypatch`, or similar;" not in agents
