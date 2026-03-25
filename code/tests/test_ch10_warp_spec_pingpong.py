from ch10.baseline_warp_spec_pingpong import BaselineWarpSpecPingPongBenchmark
from ch10.optimized_warp_spec_pingpong import OptimizedWarpSpecPingPongBenchmark


def test_warp_spec_pingpong_wrappers_report_real_pipeline_metadata() -> None:
    baseline = BaselineWarpSpecPingPongBenchmark()
    optimized = OptimizedWarpSpecPingPongBenchmark()

    assert baseline.num_stages == 1
    assert optimized.num_stages == 2

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert baseline_metrics["pipeline.num_stages"] == 1.0
    assert baseline_metrics["pipeline.consumer_warps"] == 1.0
    assert baseline_metrics["pipeline.pingpong_enabled"] == 0.0

    assert optimized_metrics["pipeline.num_stages"] == 2.0
    assert optimized_metrics["pipeline.consumer_warps"] == 2.0
    assert optimized_metrics["pipeline.pingpong_enabled"] == 1.0
