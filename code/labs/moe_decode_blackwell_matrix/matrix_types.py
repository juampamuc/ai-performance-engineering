from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


ROUTING_POLICIES = ("balanced", "sticky", "skewed")
SCHEDULE_MODES = ("dynamic", "persistent")
LAUNCH_MODES = ("eager", "cuda_graph")
DTYPE_NAMES = ("bf16", "fp16")


def _normalize_int_tuple(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    normalized = tuple(int(value) for value in values)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _normalize_choice_tuple(
    values: Sequence[str],
    *,
    allowed: Sequence[str],
    field_name: str,
) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    normalized = tuple(str(value) for value in values)
    invalid = sorted(set(normalized) - set(allowed))
    if invalid:
        raise ValueError(f"{field_name} contains unsupported values: {invalid}")
    return normalized


def classify_batch_regime(decode_batch: int) -> str:
    if decode_batch <= 4:
        return "micro"
    if decode_batch <= 32:
        return "latency"
    return "throughput"


@dataclass(frozen=True)
class MatrixScenario:
    playbook_name: str
    description: str
    seed: int
    dtype: str
    hidden_size: int
    intermediate_size: int
    steps: int
    warmup: int
    repeats: int
    num_experts: int
    top_k: int
    decode_batch: int
    routing_policy: str
    schedule_mode: str
    launch_mode: str

    @property
    def batch_regime(self) -> str:
        return classify_batch_regime(self.decode_batch)

    @property
    def workload_key(self) -> str:
        return (
            f"e{self.num_experts}_k{self.top_k}_b{self.decode_batch}"
            f"_{self.routing_policy}_{self.batch_regime}"
        )

    @property
    def config_id(self) -> str:
        route_tag = {
            "balanced": "bal",
            "sticky": "stk",
            "skewed": "skw",
        }[self.routing_policy]
        schedule_tag = {
            "dynamic": "dyn",
            "persistent": "pst",
        }[self.schedule_mode]
        launch_tag = {
            "eager": "egr",
            "cuda_graph": "grf",
        }[self.launch_mode]
        return (
            f"e{self.num_experts}_k{self.top_k}_b{self.decode_batch}"
            f"_{route_tag}_{schedule_tag}_{launch_tag}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["batch_regime"] = self.batch_regime
        payload["workload_key"] = self.workload_key
        payload["config_id"] = self.config_id
        return payload


@dataclass(frozen=True)
class MatrixPlaybook:
    name: str
    description: str
    seed: int = 17
    dtype: str = "bf16"
    hidden_size: int = 512
    intermediate_size: int = 2048
    steps: int = 8
    warmup: int = 2
    repeats: int = 6
    expert_counts: tuple[int, ...] = (8, 16, 32)
    top_k_values: tuple[int, ...] = (1, 2, 4)
    decode_batches: tuple[int, ...] = (1, 8, 32, 128)
    routing_policies: tuple[str, ...] = ("balanced", "sticky", "skewed")
    schedule_modes: tuple[str, ...] = ("dynamic", "persistent")
    launch_modes: tuple[str, ...] = ("eager", "cuda_graph")
    sm_clock_mhz: int | None = 1500
    mem_clock_mhz: int | None = None
    allow_non_blackwell: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {DTYPE_NAMES}, got {self.dtype!r}")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.steps <= 0 or self.warmup < 0 or self.repeats <= 0:
            raise ValueError("steps, warmup, and repeats must be non-negative/positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        _normalize_int_tuple(self.expert_counts, field_name="expert_counts")
        _normalize_int_tuple(self.top_k_values, field_name="top_k_values")
        _normalize_int_tuple(self.decode_batches, field_name="decode_batches")
        _normalize_choice_tuple(
            self.routing_policies,
            allowed=ROUTING_POLICIES,
            field_name="routing_policies",
        )
        _normalize_choice_tuple(
            self.schedule_modes,
            allowed=SCHEDULE_MODES,
            field_name="schedule_modes",
        )
        _normalize_choice_tuple(
            self.launch_modes,
            allowed=LAUNCH_MODES,
            field_name="launch_modes",
        )
        if max(self.top_k_values) > min(self.expert_counts):
            raise ValueError("top_k_values cannot exceed the smallest configured expert count")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatrixPlaybook":
        payload = dict(data)
        payload["expert_counts"] = _normalize_int_tuple(
            payload.get("expert_counts", cls.expert_counts), field_name="expert_counts"
        )
        payload["top_k_values"] = _normalize_int_tuple(
            payload.get("top_k_values", cls.top_k_values), field_name="top_k_values"
        )
        payload["decode_batches"] = _normalize_int_tuple(
            payload.get("decode_batches", cls.decode_batches), field_name="decode_batches"
        )
        payload["routing_policies"] = _normalize_choice_tuple(
            payload.get("routing_policies", cls.routing_policies),
            allowed=ROUTING_POLICIES,
            field_name="routing_policies",
        )
        payload["schedule_modes"] = _normalize_choice_tuple(
            payload.get("schedule_modes", cls.schedule_modes),
            allowed=SCHEDULE_MODES,
            field_name="schedule_modes",
        )
        payload["launch_modes"] = _normalize_choice_tuple(
            payload.get("launch_modes", cls.launch_modes),
            allowed=LAUNCH_MODES,
            field_name="launch_modes",
        )
        return cls(**payload)

    def with_overrides(
        self,
        *,
        expert_counts: Sequence[int] | None = None,
        top_k_values: Sequence[int] | None = None,
        decode_batches: Sequence[int] | None = None,
        routing_policies: Sequence[str] | None = None,
        schedule_modes: Sequence[str] | None = None,
        launch_modes: Sequence[str] | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        steps: int | None = None,
        warmup: int | None = None,
        repeats: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        sm_clock_mhz: int | None = None,
        mem_clock_mhz: int | None = None,
        allow_non_blackwell: bool | None = None,
    ) -> "MatrixPlaybook":
        updates: dict[str, Any] = {}
        if expert_counts is not None:
            updates["expert_counts"] = _normalize_int_tuple(
                expert_counts, field_name="expert_counts"
            )
        if top_k_values is not None:
            updates["top_k_values"] = _normalize_int_tuple(
                top_k_values, field_name="top_k_values"
            )
        if decode_batches is not None:
            updates["decode_batches"] = _normalize_int_tuple(
                decode_batches, field_name="decode_batches"
            )
        if routing_policies is not None:
            updates["routing_policies"] = _normalize_choice_tuple(
                routing_policies,
                allowed=ROUTING_POLICIES,
                field_name="routing_policies",
            )
        if schedule_modes is not None:
            updates["schedule_modes"] = _normalize_choice_tuple(
                schedule_modes,
                allowed=SCHEDULE_MODES,
                field_name="schedule_modes",
            )
        if launch_modes is not None:
            updates["launch_modes"] = _normalize_choice_tuple(
                launch_modes,
                allowed=LAUNCH_MODES,
                field_name="launch_modes",
            )
        for field_name, value in (
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("steps", steps),
            ("warmup", warmup),
            ("repeats", repeats),
            ("dtype", dtype),
            ("seed", seed),
            ("sm_clock_mhz", sm_clock_mhz),
            ("mem_clock_mhz", mem_clock_mhz),
            ("allow_non_blackwell", allow_non_blackwell),
        ):
            if value is not None:
                updates[field_name] = value
        return replace(self, **updates)

    def iter_scenarios(self) -> Iterable[MatrixScenario]:
        for (
            num_experts,
            top_k,
            decode_batch,
            routing_policy,
            schedule_mode,
            launch_mode,
        ) in product(
            self.expert_counts,
            self.top_k_values,
            self.decode_batches,
            self.routing_policies,
            self.schedule_modes,
            self.launch_modes,
        ):
            if top_k > num_experts:
                continue
            yield MatrixScenario(
                playbook_name=self.name,
                description=self.description,
                seed=self.seed,
                dtype=self.dtype,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                steps=self.steps,
                warmup=self.warmup,
                repeats=self.repeats,
                num_experts=num_experts,
                top_k=top_k,
                decode_batch=decode_batch,
                routing_policy=routing_policy,
                schedule_mode=schedule_mode,
                launch_mode=launch_mode,
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expert_counts"] = list(self.expert_counts)
        payload["top_k_values"] = list(self.top_k_values)
        payload["decode_batches"] = list(self.decode_batches)
        payload["routing_policies"] = list(self.routing_policies)
        payload["schedule_modes"] = list(self.schedule_modes)
        payload["launch_modes"] = list(self.launch_modes)
        return payload
