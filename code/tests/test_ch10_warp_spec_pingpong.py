from ch10.baseline_warp_spec_pingpong import BaselineWarpSpecPingPongBenchmark
from ch10.optimized_warp_spec_pingpong import OptimizedWarpSpecPingPongBenchmark


def test_warp_spec_pingpong_wrappers_report_real_pipeline_metadata() -> None:
    baseline = BaselineWarpSpecPingPongBenchmark()
    optimized = OptimizedWarpSpecPingPongBenchmark()

    assert baseline.num_stages == 1
    assert optimized.num_stages == 2

    assert baseline.get_custom_metrics() == {
        "pipeline.num_stages": 1.0,
        "pipeline.consumer_warps": 2.0,
        "pipeline.pingpong_enabled": 0.0,
    }
    assert optimized.get_custom_metrics() == {
        "pipeline.num_stages": 2.0,
        "pipeline.consumer_warps": 2.0,
        "pipeline.pingpong_enabled": 1.0,
    }
