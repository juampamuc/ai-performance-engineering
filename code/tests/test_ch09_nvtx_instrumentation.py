from pathlib import Path


def test_fp4_perchannel_sources_do_not_emit_hot_loop_nvtx_ranges() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "ch09" / "baseline_cublas_gemm_fp4_perchannel.cu",
        repo_root / "ch09" / "optimized_cublas_gemm_fp4_perchannel.cu",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert 'NVTX_RANGE("iteration")' not in text
        assert 'NVTX_RANGE("verify")' not in text


def test_cutlass_fp4_perchannel_sources_hoist_setup_out_of_hot_loop() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "ch09" / "baseline_cutlass_gemm_fp4_perchannel.cu",
        repo_root / "ch09" / "optimized_cutlass_gemm_fp4_perchannel.cu",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert 'NVTX_RANGE("setup")' not in text
        assert text.count("gemm.initialize(arguments, workspace.get())") == 1


def test_optimized_cutlass_fp4_perchannel_uses_fused_epilogue_not_post_scale_kernel() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline = (repo_root / "ch09" / "baseline_cutlass_gemm_fp4_perchannel.cu").read_text(encoding="utf-8")
    optimized = (repo_root / "ch09" / "optimized_cutlass_gemm_fp4_perchannel.cu").read_text(encoding="utf-8")

    assert "apply_per_channel_scale<<<" in baseline
    assert "PerColLinCombPerColBiasEltAct" in optimized
    assert "apply_per_channel_scale<<<" not in optimized
    assert "alpha_ptr = d_scales" in optimized


def test_optimized_cutlass_fp4_perchannel_fuses_scale_in_epilogue() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "ch09" / "optimized_cutlass_gemm_fp4_perchannel.cu").read_text(
        encoding="utf-8"
    )

    assert "PerColLinCombPerColBiasEltAct" in text
    assert "fusion_args.alpha_ptr = d_scales" in text
    assert "CUTLASS_CHECK(gemm.run(stream));" in text
    assert "apply_per_channel_scale<<<" not in text
