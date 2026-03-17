"""Optimized multi-node aware DeepSeek-style hybrid expert-parallel benchmark."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark

from labs.fullstack_cluster.moe_hybrid_ep_common import MoEHybridEPBenchmark, run_cli


def main() -> None:
    run_cli(optimized=True)


def get_benchmark() -> BaseBenchmark:
    bench = MoEHybridEPBenchmark(
        optimized=True,
        multigpu=True,
        script_path=__file__,
        label="optimized_moe_hybrid_ep_multigpu",
    )
    return attach_benchmark_metadata(bench, __file__)


def _maybe_run_cli() -> None:
    if globals().get("__name__") != "__main__":
        return
    main()


_maybe_run_cli()
