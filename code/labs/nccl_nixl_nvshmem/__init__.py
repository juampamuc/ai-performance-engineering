"""Lab for communication-stack tradeoffs across NCCL, NIXL, and NVSHMEM."""

from .comm_stack_common import (
    TierHandoffWorkload,
    default_workload,
    probe_communication_stack,
)

__all__ = [
    "TierHandoffWorkload",
    "default_workload",
    "probe_communication_stack",
]
