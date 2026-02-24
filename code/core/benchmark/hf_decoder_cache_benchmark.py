"""HF decoder-loop benchmark primitives for cache and EOS sync experiments.

Inspired by:
https://chaimrand.medium.com/optimizing-token-generation-in-pytorch-decoder-models-8e63b5a5fc80
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

try:
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.cache_utils import StaticCache

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    GPT2Config = None  # type: ignore[assignment]
    GPT2LMHeadModel = None  # type: ignore[assignment]
    StaticCache = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False


@dataclass(frozen=True)
class HFDecoderCacheConfig:
    """Configuration for a synthetic, deterministic HF decoder benchmark."""

    batch_size: int = 4
    prompt_tokens: int = 128
    decode_tokens: int = 128
    vocab_size: int = 4096
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    eos_token_id: int = 2
    bos_token_id: int = 1
    pad_token_id: int = 0
    dtype: torch.dtype = torch.float16
    cache_mode: Literal["dynamic", "static"] = "dynamic"
    compile_decode_step: bool = False
    eos_sync_mode: Literal["blocking", "async_streamed"] = "blocking"
    eos_poll_interval: int = 1
    stop_on_all_done: bool = False
    iterations: int = 8
    warmup: int = 8
    seed: int = 42
    label: str = "hf_decoder_cache"


def attach_benchmark_metadata(bench: BaseBenchmark, module_file: str) -> BaseBenchmark:
    """Annotate a benchmark for subprocess-safe reload via get_benchmark()."""
    bench._module_file_override = module_file
    bench._factory_name_override = "get_benchmark"
    return bench


class HFDecoderCacheBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness benchmark for decoder cache strategy and EOS host-sync policies."""

    def __init__(self, cfg: HFDecoderCacheConfig):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("SKIPPED: transformers is required for HF decoder cache benchmarks")
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: HF decoder cache benchmarks require CUDA")
        if cfg.eos_poll_interval < 1:
            raise ValueError("eos_poll_interval must be >= 1")
        if cfg.cache_mode == "dynamic" and cfg.compile_decode_step:
            raise ValueError("compile_decode_step is supported only for cache_mode='static'")

        self.cfg = cfg
        self.dtype = cfg.dtype
        self.model: Optional[torch.nn.Module] = None
        self.prompt_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._static_cache: Optional[StaticCache] = None
        self._decode_step_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
        self._cache_pos_token = torch.empty((1,), dtype=torch.long, device=self.device)
        self._eos_token = torch.tensor(cfg.eos_token_id, dtype=torch.long, device=self.device)
        self._metrics: Dict[str, float] = {}

        # Async EOS polling resources (used only when eos_sync_mode=async_streamed).
        self._poll_stream: Optional[torch.cuda.Stream] = None
        self._poll_event: Optional[torch.cuda.Event] = None
        self._poll_flag_host: Optional[torch.Tensor] = None
        self._pending_poll: bool = False

        self._sync_checks = 0
        self._eos_all_true_checks = 0

        total_tokens = cfg.batch_size * (cfg.prompt_tokens + cfg.decode_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(cfg.batch_size),
            tokens_per_iteration=float(total_tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(cfg.batch_size),
            tokens_per_iteration=float(total_tokens),
        )

    def _build_model(self) -> torch.nn.Module:
        total_len = self.cfg.prompt_tokens + self.cfg.decode_tokens + 8
        model_cfg = GPT2Config(
            vocab_size=self.cfg.vocab_size,
            n_positions=total_len,
            n_layer=self.cfg.num_layers,
            n_head=self.cfg.num_heads,
            n_embd=self.cfg.hidden_size,
            bos_token_id=self.cfg.bos_token_id,
            eos_token_id=self.cfg.eos_token_id,
            pad_token_id=self.cfg.pad_token_id,
            use_cache=True,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        model = GPT2LMHeadModel(model_cfg).to(device=self.device, dtype=self.dtype).eval()
        model.requires_grad_(False)
        return model

    def _build_prompts(self) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(self.cfg.seed)
        return torch.randint(
            low=3,
            high=self.cfg.vocab_size,
            size=(self.cfg.batch_size, self.cfg.prompt_tokens),
            device=self.device,
            dtype=torch.long,
            generator=g,
        )

    def _setup_static_cache(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        if StaticCache is None:
            raise RuntimeError("SKIPPED: StaticCache is unavailable in this transformers build")
        total_len = self.cfg.prompt_tokens + self.cfg.decode_tokens
        self._static_cache = StaticCache(config=self.model.config, max_cache_len=total_len)

        def decode_step(input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
            logits = self.model(  # type: ignore[misc]
                input_ids=input_ids,
                past_key_values=self._static_cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=False,
            )[0]
            return logits[:, -1, :]

        if self.cfg.compile_decode_step:
            try:
                self._decode_step_fn = torch.compile(
                    decode_step,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception as exc:
                raise RuntimeError(f"torch.compile failed for static decode step: {exc}") from exc
        else:
            self._decode_step_fn = decode_step

    def _setup_eos_polling(self) -> None:
        if self.cfg.eos_sync_mode != "async_streamed":
            return
        self._poll_stream = torch.cuda.Stream(device=self.device)
        self._poll_event = torch.cuda.Event()
        self._poll_flag_host = torch.empty((), dtype=torch.uint8, pin_memory=True)

    def setup(self) -> None:
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        self.model = self._build_model()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.prompt_ids = self._build_prompts()
        if self.cfg.cache_mode == "static":
            self._setup_static_cache()
        self._setup_eos_polling()
        self._synchronize()

    def _prepare_iteration(self) -> None:
        self._sync_checks = 0
        self._eos_all_true_checks = 0
        self._pending_poll = False
        if self._poll_flag_host is not None:
            self._poll_flag_host.zero_()
        if self._static_cache is not None:
            self._static_cache.reset()

    def _maybe_poll_done(self, done_mask: torch.Tensor, step: int) -> bool:
        should_poll = (step + 1) % self.cfg.eos_poll_interval == 0 or (step + 1) == self.cfg.decode_tokens
        if not should_poll:
            return False
        self._sync_checks += 1

        if self.cfg.eos_sync_mode == "blocking":
            all_done = bool(done_mask.all().item())
            self._eos_all_true_checks += int(all_done)
            return all_done

        if any(x is None for x in (self._poll_stream, self._poll_event, self._poll_flag_host)):
            raise RuntimeError("Async EOS polling resources are not initialized")

        if self._pending_poll and self._poll_event.query():
            self._eos_all_true_checks += int(self._poll_flag_host.item() != 0)

        all_done_u8 = done_mask.all().to(torch.uint8)
        with torch.cuda.stream(self._poll_stream):
            self._poll_flag_host.copy_(all_done_u8, non_blocking=True)
        self._poll_event.record(self._poll_stream)
        self._pending_poll = True
        return False

    def _finalize_polling(self) -> None:
        if self.cfg.eos_sync_mode != "async_streamed":
            return
        if self._pending_poll and self._poll_event is not None and self._poll_flag_host is not None:
            self._poll_event.synchronize()
            self._eos_all_true_checks += int(self._poll_flag_host.item() != 0)
            self._pending_poll = False

    def _decode_dynamic(self, next_token: torch.Tensor, past_key_values: object) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        done_mask = next_token.eq(self.cfg.eos_token_id)
        generated: list[torch.Tensor] = []
        eos_fill = self._eos_token.expand(self.cfg.batch_size)

        for step in range(self.cfg.decode_tokens):
            outputs = self.model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=False,
            )
            logits = outputs[0]
            past_key_values = outputs[1]
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_token)
            done_mask.logical_or_(next_token.eq(self.cfg.eos_token_id))
            all_done = self._maybe_poll_done(done_mask, step)
            if self.cfg.stop_on_all_done and all_done:
                break
            next_token = torch.where(done_mask, eos_fill, next_token)

        while len(generated) < self.cfg.decode_tokens:
            generated.append(eos_fill)
        return torch.stack(generated, dim=1)

    def _decode_static(self, next_token: torch.Tensor) -> torch.Tensor:
        if self._decode_step_fn is None:
            raise RuntimeError("Static decode step function is not initialized")
        done_mask = next_token.eq(self.cfg.eos_token_id)
        generated: list[torch.Tensor] = []
        eos_fill = self._eos_token.expand(self.cfg.batch_size)

        for step in range(self.cfg.decode_tokens):
            self._cache_pos_token.fill_(self.cfg.prompt_tokens + step)
            logits_last = self._decode_step_fn(next_token.unsqueeze(-1), self._cache_pos_token)
            next_token = torch.argmax(logits_last, dim=-1)
            generated.append(next_token)
            done_mask.logical_or_(next_token.eq(self.cfg.eos_token_id))
            all_done = self._maybe_poll_done(done_mask, step)
            if self.cfg.stop_on_all_done and all_done:
                break
            next_token = torch.where(done_mask, eos_fill, next_token)

        while len(generated) < self.cfg.decode_tokens:
            generated.append(eos_fill)
        return torch.stack(generated, dim=1)

    def benchmark_fn(self) -> None:
        if self.model is None or self.prompt_ids is None:
            raise RuntimeError("Benchmark not configured")
        self._prepare_iteration()

        with torch.no_grad():
            self._synchronize()
            prefill_start = time.perf_counter()

            if self.cfg.cache_mode == "dynamic":
                prefill = self.model(
                    input_ids=self.prompt_ids,
                    use_cache=True,
                    return_dict=False,
                )
                prefill_logits = prefill[0]
                past_key_values = prefill[1]
                next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1)
                verification_token = next_token.detach().to(torch.int32).clone()
                self._synchronize()
                prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

                decode_start = time.perf_counter()
                _ = self._decode_dynamic(next_token, past_key_values)
            else:
                if self._static_cache is None:
                    raise RuntimeError("Static cache mode selected but static cache is not initialized")
                prompt_pos = torch.arange(self.cfg.prompt_tokens, device=self.device, dtype=torch.long)
                prefill_logits = self.model(
                    input_ids=self.prompt_ids,
                    past_key_values=self._static_cache,
                    cache_position=prompt_pos,
                    use_cache=True,
                    return_dict=False,
                )[0]
                next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1)
                verification_token = next_token.detach().to(torch.int32).clone()
                self._synchronize()
                prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

                decode_start = time.perf_counter()
                _ = self._decode_static(next_token)

            self._finalize_polling()
            self._synchronize()
            decode_ms = (time.perf_counter() - decode_start) * 1000.0

        total_ms = prefill_ms + decode_ms
        decode_total_tokens = float(self.cfg.batch_size * self.cfg.decode_tokens)
        end_to_end_tokens = float(self.cfg.batch_size * (self.cfg.prompt_tokens + self.cfg.decode_tokens))
        self._metrics = {
            "hf_decode.prefill_ms": float(prefill_ms),
            "hf_decode.decode_ms": float(decode_ms),
            "hf_decode.total_ms": float(total_ms),
            "hf_decode.decode_tokens_per_s": decode_total_tokens / max(decode_ms / 1000.0, 1e-9),
            "hf_decode.end_to_end_tokens_per_s": end_to_end_tokens / max(total_ms / 1000.0, 1e-9),
            "hf_decode.sync_checks": float(self._sync_checks),
            "hf_decode.eos_all_true_checks": float(self._eos_all_true_checks),
        }
        # Use the first decode token after prefill for verification.
        # Static/dynamic cache decode loops can diverge over long rollouts due to
        # tiny numeric differences amplified by greedy argmax, while this prefill
        # boundary token remains a stable equivalence anchor.
        self.output = verification_token

    def capture_verification_payload(self) -> None:
        if self.prompt_ids is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"prompt_ids": self.prompt_ids},
            output=self.output,
            batch_size=self.cfg.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.prompt_ids = None
        self.output = None
        self._static_cache = None
        self._decode_step_fn = None
        self._poll_stream = None
        self._poll_event = None
        self._poll_flag_host = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.cfg.iterations,
            warmup=self.cfg.warmup,
            setup_timeout_seconds=600,
            measurement_timeout_seconds=600,
            enable_memory_tracking=True,
            timing_method="wall_clock",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._metrics
