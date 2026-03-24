from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_ch15_optimized_dep2_parallel_fails_fast_when_compile_breaks() -> None:
    source = _read("ch15/optimized_dep2_parallel.py")
    assert "FAIL FAST: torch.compile failed for optimized_dep2_parallel" in source
    assert "self._compiled_fn = _run" not in source


def test_ch15_disaggregated_inference_no_longer_swallows_sdpa_backend_failures() -> None:
    source = _read("ch15/disaggregated_inference_multigpu.py")
    assert "falling back gracefully" not in source
    assert "return nullcontext()" not in source
    assert "return sdpa_kernel(list(PREFERRED_SDP_BACKENDS))" in source


def test_ch18_paged_attention_sparse_path_fails_fast_on_compile_errors() -> None:
    source = _read("ch18/paged_attn_split_common.py")
    assert "FAIL FAST: torch.compile(flex_attention) failed for paged attention sparse path" in source
    assert "return flex_attention" not in source


def test_ch18_nvfp4_trtllm_tool_no_longer_uses_placeholder_outputs_or_eager_fallback() -> None:
    source = _read("ch18/nvfp4_trtllm_tool.py")
    assert "torch.tensor([float(len(outputs))]" not in source
    assert "FAIL FAST: TRT-LLM generate returned an unsupported output payload" in source
    assert "FAIL FAST: Transformer Engine FP8 path failed in nvfp4_trtllm_tool" in source
