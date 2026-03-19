# Lab - Blackwell Grouped GEMM Optimizations

## Summary
Implements the grouped/MoE GEMM optimization journey from the Blackwell slide deck inside the harness: start with a small-tile grouped kernel, then move to larger tiles, fused route-weight application, `GROUP_SIZE_M` swizzling, autotune, and a persistent tile schedule.

## Targets
| Target | Stage |
| --- | --- |
| `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm` | canonical baseline plus all optimized siblings |
| `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles` | Step 1: `128x128x64` tiles |
| `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack` | Step 3: fused route weights + `GROUP_SIZE_M` + autotune |
| `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent` | Step 4: persistent strided tile loop |

## Problem
The source deck is not a plain dense GEMM story. It is a routed/grouped GEMM story where the useful question is: once tokens are bucketed per expert, which schedule choices actually move grouped expert throughput on Blackwell?

This lab keeps that shape honest:
- tokens are packed into per-expert buckets
- route weights stay visible in the workload
- the public targets map to the real optimization stages instead of collapsing everything into one opaque "optimized" kernel

## Baseline Path
- `64x64x32` tiles
- one grouped tile launch model without persistent residency
- route weights applied after the grouped GEMM instead of inside the kernel
- useful control for launch/schedule overhead, but not a realistic Blackwell endpoint

## Optimized Path
- `large_tiles`: `128x128x64` grouped kernel to improve tensor-core utilization
- `full_stack`: fused route-weight multiply, `GROUP_SIZE_M` swizzle, and autotuned schedule selection
- `persistent`: full-stack kernel with a strided tile loop over the total grouped tile space

## Negative Controls
The deck also includes changes that are useful to measure but should not become public default targets:

- `fast_math_control`: represented as a control only, because the Triton path has no separate user-visible fast-math toggle
- `latency10`: deeper buffering / stage count increase
- `two_cta`: extra resident warp pressure without a stable win
- `tile_n256`: larger N tile that increases working-set pressure

Use `python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --include-experimental` when you want those controls, but keep the public harness surface on the three real milestones.

## Measured Delta
Current implementation-state measurements from March 19, 2026 on the local virtualized B200 host:

| Measurement | Shape / Artifact | Result | What it proves |
| --- | --- | --- | --- |
| Direct matrix smoke (`artifacts/blackwell_grouped_gemm_smoke_matrix.csv`) | `tokens=128, hidden=256, ffn=512` | `baseline=0.206 ms`, `large_tiles=0.158 ms`, `full_stack=0.176 ms`, `persistent=0.165 ms` | all three public optimized stages beat the baseline on a tiny grouped workload, and the negative controls remain runnable for comparison |
| Harness pair (`artifacts/runs/20260319_015359__bench__profile_minimal_targets_labs_blackwell_gemm_optimizations_blackwell_grouped_gemm/`) | `tokens=512, hidden=512, ffn=1024` | `baseline=0.132 ms`, `persistent=0.131 ms`, `1.01x` | the public persistent pair already runs with locked clocks, verification, and Nsight traces |

These numbers are directional, not canonical: the current host is virtualized, and the smoke matrix is intentionally small so implementation bugs show up quickly. Treat the tiny-shape ranking as a functional smoke check, not as the final promotion signal. The important point is that the grouped schedule transitions are now reproducible inside the repo instead of living only in screenshots.

## Profiler Evidence
Use deep-dive runs when you want Nsight evidence for the grouped schedule transitions:

```bash
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles --profile deep_dive --single-gpu --gpu-sm-clock-mhz 1500
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack --profile deep_dive --single-gpu --gpu-sm-clock-mhz 1500
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile deep_dive --single-gpu --gpu-sm-clock-mhz 1500
```

The implementation-state persistent pair already has `nsys`, `ncu`, and PyTorch traces in `artifacts/runs/20260319_015359__bench__profile_minimal_targets_labs_blackwell_gemm_optimizations_blackwell_grouped_gemm/`.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/blackwell_gemm_optimizations
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --token-counts 1024 2048 4096 8192 --include-experimental
```

## Learning Goals
- Keep the Blackwell deck honest by exposing grouped/MoE GEMM rather than pretending the story is dense matmul.
- Show which gains come from larger tiles versus fusion/swizzle versus persistent residency.
- Preserve the negative controls as runnable experiments without promoting them to public benchmark targets.
- Give the repo a Blackwell-gated grouped GEMM lab that can be profiled, verified, and extended with future kernel work.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_blackwell_grouped_gemm.py`, `optimized_blackwell_grouped_gemm_large_tiles.py`, `optimized_blackwell_grouped_gemm_full_stack.py`, `optimized_blackwell_grouped_gemm_persistent.py` | Public harness wrappers for the baseline and three optimization stages. |
| `blackwell_grouped_gemm_common.py` | Deterministic grouped-workload builder, BF16/FP16 reference path, benchmark class, and verification logic. |
| `blackwell_grouped_gemm_kernel.py`, `blackwell_grouped_gemm_autotune.py` | Triton kernel launches, schedule registry, autotune candidates, and experimental controls. |
| `compare_blackwell_grouped_gemm_matrix.py` | Direct token-count sweep runner for smoke checks and negative-control comparisons outside the harness. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/blackwell_gemm_optimizations
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --token-counts 1024 2048 4096 8192 --include-experimental
```
- Use `--target-extra-arg labs/blackwell_gemm_optimizations:<workload>="--num-tokens N --hidden-dim K --expert-ffn-dim N"` to sweep grouped-token shapes through the harness.
- Launch the direct runner as a module (`python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix`), not as a bare script path.
- The direct runner is a smoke/perf helper. Use the harnessed `bench run` targets for clock-locked pair results and profiler artifacts.

## Validation Checklist
- `python -m cli.aisp bench list-targets --chapter labs/blackwell_gemm_optimizations` exposes the canonical baseline target plus the `large_tiles`, `full_stack`, and `persistent` aliases.
- `python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile minimal --single-gpu --gpu-sm-clock-mhz 1500 --iterations 2 --warmup 1 --target-extra-arg 'labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent=--num-tokens 512 --hidden-dim 512 --expert-ffn-dim 1024'` succeeds with verification and produces `nsys`, `ncu`, and PyTorch traces under `artifacts/runs/20260319_015359__bench__profile_minimal_targets_labs_blackwell_gemm_optimizations_blackwell_grouped_gemm/`.
- `python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --token-counts 128 --num-experts 4 --hidden-dim 256 --expert-ffn-dim 512 --warmup 1 --repeats 2` shows the expected stage-wise speedup trend on the implementation smoke shape.
- `pytest tests/test_blackwell_gemm_optimizations_lab.py` validates discovery, wrapper exposure, and per-variant CUDA parity on a small grouped workload.

## Notes
- This lab is intentionally grouped/MoE-flavored GEMM. The directory name keeps the user-facing Blackwell GEMM framing, but the benchmark targets stay explicit about grouped execution.
- The slide-deck Step 2 fast-math idea is preserved only as a control because the Triton path does not expose an equivalent standalone toggle.
