import torch

from ch15.inference_monolithic_common import SimpleLLM


def test_decode_autoregressive_matches_decode_output() -> None:
    model = SimpleLLM(vocab_size=128, hidden_dim=32, num_layers=3).eval()
    prompt = torch.arange(8, dtype=torch.int64).unsqueeze(0) % 128
    kv_cache = model.prefill(prompt)
    direct = model.decode(kv_cache, num_tokens=6)
    output_buffer = torch.empty(1, 6, model.hidden_dim, dtype=kv_cache.dtype)
    reused = model.decode_autoregressive(kv_cache, num_tokens=6, output_buffer=output_buffer)
    assert reused.data_ptr() == output_buffer.data_ptr()
    assert torch.allclose(direct, reused)


def test_decode_autoregressive_rejects_wrong_buffer_shape() -> None:
    model = SimpleLLM(vocab_size=64, hidden_dim=16, num_layers=2).eval()
    prompt = torch.arange(4, dtype=torch.int64).unsqueeze(0) % 64
    kv_cache = model.prefill(prompt)
    bad_buffer = torch.empty(1, 5, model.hidden_dim, dtype=kv_cache.dtype)
    try:
        model.decode_autoregressive(kv_cache, num_tokens=4, output_buffer=bad_buffer)
    except ValueError as exc:
        assert "output_buffer shape" in str(exc)
    else:
        raise AssertionError("decode_autoregressive() should reject mismatched output buffers")
