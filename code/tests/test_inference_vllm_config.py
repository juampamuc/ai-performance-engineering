import json


def test_engine_vllm_config_is_json_serializable() -> None:
    from core.engine import get_engine

    result = get_engine().inference.vllm_config(
        model="meta-llama/Llama-3.1-8B",
        model_params_b=7,
        num_gpus=1,
        gpu_memory_gb=80,
        target="throughput",
        max_seq_length=8192,
        compare=False,
    )

    assert result["success"] is True
    config = result["vllm_config"]
    assert isinstance(config, dict)
    assert config["tensor_parallel_size"] >= 1
    assert config["max_num_seqs"] >= 1
    assert config["max_num_batched_tokens"] >= 1
    json.dumps(result)


def test_engine_vllm_compare_is_json_serializable() -> None:
    from core.engine import get_engine

    result = get_engine().inference.vllm_config(
        model="meta-llama/Llama-3.1-8B",
        model_params_b=7,
        num_gpus=1,
        compare=True,
    )

    assert result["success"] is True
    assert result["engine_comparison"]["recommended"]
    json.dumps(result)
