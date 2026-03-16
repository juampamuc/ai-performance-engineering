"""Shared helpers for Chapter 10 FlashAttention benchmarks."""

from __future__ import annotations


def compute_attention_workload_metrics(
    *,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    is_causal: bool,
) -> dict:
    """Return metrics derived from the actual attention workload."""
    head_dim = hidden_dim // max(num_heads, 1)
    score_elements = batch_size * num_heads * seq_len * seq_len
    return {
        "attention.batch_size": float(batch_size),
        "attention.seq_len": float(seq_len),
        "attention.hidden_dim": float(hidden_dim),
        "attention.num_heads": float(num_heads),
        "attention.head_dim": float(head_dim),
        "attention.score_matrix_elements": float(score_elements),
        "attention.output_elements": float(batch_size * seq_len * hidden_dim),
        "attention.is_causal": 1.0 if is_causal else 0.0,
    }


_ATTENTION_BACKEND_IDS = {
    "unknown": 0.0,
    "flash_attention": 1.0,
    "efficient_attention": 2.0,
    "math": 3.0,
    "flash_attn": 4.0,
    "flash_attn_3": 5.0,
}


_ATTENTION_ENGINE_IDS = {
    "unknown": 0.0,
    "sdpa": 1.0,
    "flash_attn": 2.0,
    "flash_attn_3": 3.0,
}


def compute_attention_backend_metrics(*, engine: str, selected_backend: str) -> dict[str, float]:
    """Return structured metrics describing the actual attention engine and backend."""
    engine_name = (engine or "unknown").lower()
    backend = (selected_backend or "unknown").lower()
    uses_flash_kernel = backend in {"flash_attention", "flash_attn", "flash_attn_3"}
    return {
        "attention.engine_id": _ATTENTION_ENGINE_IDS.get(engine_name, 0.0),
        "attention.engine_is_sdpa": 1.0 if engine_name == "sdpa" else 0.0,
        "attention.engine_is_flash_attn": 1.0 if engine_name == "flash_attn" else 0.0,
        "attention.engine_is_flash_attn_3": 1.0 if engine_name == "flash_attn_3" else 0.0,
        "attention.selected_backend_id": _ATTENTION_BACKEND_IDS.get(backend, 0.0),
        "attention.selected_backend_is_flash": 1.0 if uses_flash_kernel else 0.0,
        "attention.selected_backend_is_fallback": 0.0 if uses_flash_kernel else 1.0,
    }


def compute_sdpa_backend_metrics(selected_backend: str) -> dict[str, float]:
    """Backward-compatible helper for SDPA-only callers."""
    return compute_attention_backend_metrics(engine="sdpa", selected_backend=selected_backend)
