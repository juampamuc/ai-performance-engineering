# Lab - Memory Bandwidth Patterns

## Summary
This lab turns the GTC deck `Maximize Memory Bandwidth on Modern GPUs: Practical Techniques, Patterns, and Working Examples` into a small set of measured kernels instead of a slide recap. The main benchmark pair stays narrow:

- baseline: naive transpose with coalesced reads and strided writes
- optimized: shared-memory tiled transpose that restores coalesced writes

The standalone runner adds three supporting milestones that make the deck themes measurable on the same input:

- contiguous scalar copy
- contiguous vectorized copy
- cp.async-style double-buffered copy

That gives one honest benchmark pair plus a few extra checkpoints that explain why the pair behaves the way it does.

## Source
- Deck: `.cursor/plans/NVIDIA GTC 2026 - Maximize Memory Bandwidth on Modern GPUs_ Practical Techniques, Patterns, and Working Examples.pdf`

## Problem
This lab answers two concrete questions:

1. How much bandwidth do we lose when a memory pattern turns coalesced traffic into strided traffic?
2. On this machine, which recovery tactics actually move the needle: vectorized loads, shared-memory staging, or cp.async-style double buffering?

The lab deliberately does not claim that every advanced feature wins. The async milestone is included so you can measure it, not assume it is the best path.

## Baseline Path
- `transpose_naive`
- row-major reads stay coalesced
- transposed writes become strided and waste bandwidth

## Optimized Path
- `transpose_tiled`
- stage a tile through shared memory
- keep both the load side and the store side coalesced

## Extra Milestones
- `copy_scalar`: contiguous copy roofline reference
- `copy_vectorized`: `float4` copy to show instruction-level vectorization on a contiguous path
- `copy_async_double_buffered`: cp.async-style block copy with double-buffered shared-memory stages

## What The Milestones Mean
- `copy_scalar` vs `copy_vectorized` isolates whether wider loads help fill the bus on a contiguous pattern.
- `transpose_naive` vs `transpose_tiled` isolates the strided-write penalty and the shared-memory fix.
- `copy_async_double_buffered` shows whether extra bytes-in-flight help on this size and GPU. It may win, tie, or regress. That is the point of measuring it directly.

## Hardware Gating
- The lab requires CUDA.
- The main benchmark pair works on any CUDA GPU that can build the extension.
- The async milestone requires cp.async-capable hardware (`sm80+`). If you request `copy_async_double_buffered` on unsupported hardware, the runner raises immediately. It does not silently skip or downgrade.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_bandwidth_patterns.py` | Harness baseline: naive transpose with strided writes. |
| `optimized_bandwidth_patterns.py` | Harness optimized path: shared-memory tiled transpose. |
| `bandwidth_patterns_common.py` | Shared config, measurement helpers, and benchmark classes. |
| `bandwidth_patterns_extension.py` | JIT loader for the local CUDA extension. |
| `bandwidth_patterns_kernels.cu` | Copy and transpose kernels used by the lab. |
| `compare_bandwidth_patterns.py` | Standalone runner for the milestone table and JSON artifact. |

## Repro Commands
```bash
python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --rows 4096 --cols 8192 --json-out /tmp/memory_bandwidth_patterns.json
python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --variants copy_scalar,copy_vectorized,transpose_naive,transpose_tiled
python -m cli.aisp bench list-targets --chapter labs/memory_bandwidth_patterns
python -m cli.aisp bench run --targets labs/memory_bandwidth_patterns:bandwidth_patterns --profile none --single-gpu
```

## Running The Lab
Use the standalone runner when you want the milestone breakdown:
```bash
python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py
python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --rows 4096 --cols 8192 --warmup 5 --iterations 10
python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --json --json-out /tmp/memory_bandwidth_patterns.json
```

Use the harness target when you want the repo-native baseline/optimized pair:
```bash
python -m cli.aisp bench run --targets labs/memory_bandwidth_patterns:bandwidth_patterns --profile none --single-gpu
python -m cli.aisp bench run \
  --targets labs/memory_bandwidth_patterns:bandwidth_patterns \
  --profile none \
  --single-gpu \
  --target-extra-arg 'labs/memory_bandwidth_patterns:bandwidth_patterns=--rows 1024 --cols 1024'
```

Notes:
- The standalone runner locks GPU clocks through the harness helper by default.
- The default standalone milestone set includes the async copy path. If you want the portable subset only, pass `--variants copy_scalar,copy_vectorized,transpose_naive,transpose_tiled`.
- The async path is intentionally opt-in by variant selection, not by silent runtime downgrade.

## Validation Checklist
- `python labs/memory_bandwidth_patterns/compare_bandwidth_patterns.py --rows 1024 --cols 1024 ...` verifies all requested variants against `torch.clone()` or `torch.transpose()`.
- `python -m cli.aisp bench run --targets labs/memory_bandwidth_patterns:bandwidth_patterns ...` runs the benchmark pair through the repo harness with clock locking and verification enabled.
- The optimized pair must stay numerically exact. This lab uses `float32` copy/transpose kernels and checks with `rtol=0.0`, `atol=0.0`.
- The async milestone must fail fast on unsupported GPUs instead of silently falling back to a scalar or synchronous copy.

## Notes
- The default shape is `4096 x 8192` so the runner moves enough data to act like a bandwidth exercise rather than a launch-overhead demo.
- The async milestone is not treated as the canonical optimized result for the pair. The canonical pair is still the transpose story: strided writes vs shared-memory recovery.
