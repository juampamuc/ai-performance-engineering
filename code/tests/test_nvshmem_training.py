from contextlib import contextmanager
import os

import pytest


_DISTRIBUTED_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@contextmanager
def _cleared_distributed_env():
    saved = {key: os.environ.get(key) for key in _DISTRIBUTED_ENV_KEYS}
    for key in _DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_nvshmem_training_example_requires_real_distributed_launch() -> None:
    with _cleared_distributed_env():
        from ch04.nvshmem_training_example import init_process_group

        with pytest.raises(RuntimeError, match="SKIPPED: .*requires torchrun/distributed launch context"):
            init_process_group()


def test_nvshmem_training_patterns_requires_real_distributed_launch() -> None:
    with _cleared_distributed_env():
        from ch04.nvshmem_training_patterns import init_process_group

        with pytest.raises(RuntimeError, match="SKIPPED: .*requires torchrun/distributed launch context"):
            init_process_group()
