from __future__ import annotations

from types import SimpleNamespace

import labs.moe_cuda_ptx.moe_cuda_ptx_common as moe_common


def test_run_layer_cuda_forward_skips_standalone_quantize_roundtrip(monkeypatch) -> None:
    workload = moe_common.MoECudaPtxWorkload(mode="forward")
    sentinel_packed = SimpleNamespace(packed_tokens="packed_tokens")
    calls = {"pack": 0, "grouped": 0, "combine": 0}

    def _pack_topk_routes(*args, **kwargs):
        calls["pack"] += 1
        return sentinel_packed

    def _grouped_ffn_cuda(
        packed_tokens,
        packed,
        gate_proj,
        up_proj,
        down_proj,
        *,
        padded_tokens_buffer=None,
    ):
        calls["grouped"] += 1
        assert packed_tokens == "packed_tokens"
        assert packed is sentinel_packed
        return "sorted_outputs"

    def _combine_weighted_outputs(sorted_outputs, packed, num_tokens, *, output_buffer=None):
        calls["combine"] += 1
        assert sorted_outputs == "sorted_outputs"
        assert packed is sentinel_packed
        assert num_tokens == workload.num_tokens
        return "combined_outputs"

    monkeypatch.setattr(moe_common, "pack_topk_routes", _pack_topk_routes)
    monkeypatch.setattr(moe_common, "grouped_ffn_cuda", _grouped_ffn_cuda)
    monkeypatch.setattr(moe_common, "combine_weighted_outputs", _combine_weighted_outputs)
    monkeypatch.setattr(
        moe_common,
        "quantize_mxfp8_optimized",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected quantize path")),
    )
    monkeypatch.setattr(
        moe_common,
        "dequantize_mxfp8",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected dequantize path")),
    )

    state = SimpleNamespace(
        x="x",
        expert_indices="expert_indices",
        expert_weights="expert_weights",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
    )

    result = moe_common.run_layer_cuda(state, workload)

    assert result == "combined_outputs"
    assert calls == {"pack": 1, "grouped": 1, "combine": 1}
