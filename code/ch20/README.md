# Chapter 20 - AI-Assisted Performance Optimization & Case Studies

## Summary
Combines AI-assisted kernel exploration with end-to-end case studies: prototype or verify generated kernels, then test whether memory, pipeline, and inference optimizations still hold up once they are composed into a full workload.

## Problem
Chapter 20 is where AI-generated ideas and isolated wins have to survive contact with the full stack. The useful question is not only "did one optimization help in isolation?" but also "can we validate generated or staged optimizations once memory, pipeline, and inference changes are stacked together in one end-to-end workload?"

## Baseline Path
- sequential or minimally optimized end-to-end execution
- independent subsystems with little cross-stage coordination
- useful as a proof baseline, but usually leaves bandwidth and overlap on the table

## Optimized Path
- staged pipeline, memory, and KV-cache optimizations combined into one workload
- the same harness contract as every other chapter, so the end-to-end gains stay comparable to the lower-level chapters
- better for answering whether the optimizations compose cleanly instead of fighting each other

## Measured Delta
Representative validated results:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `integrated_kv_cache` | `986.812 ms` | `139.676 ms` | `7.07x` | block-wise paged KV-cache integration across the end-to-end loop |
| `bf16_mlp` | `0.616 ms` | `0.234 ms` | `2.63x` | BF16 precision policy on the same eager MLP graph |

The current `integrated_kv_cache` proof point comes from `artifacts/runs/20260328_162954__bench__profile_minimal_targets_ch20_integrated_kv_cache/`. The `bf16_mlp` row remains from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`.

This chapter is the best place to check whether wins compose. `pipeline_sequential` now remains available as an informational overlap demo, while canonical chapter claims focus on the pairs that still hold up as end-to-end improvements. The AI-assisted kernel-generation thread is represented here by `ai_kernel_generator.py` plus the verifier helpers, even though the full manuscript chapter covers a broader RL/AlphaTensor narrative than the current harness surface.

## Profiler Evidence
Use deep-dive harness runs when you want to see how the end-to-end gain breaks down by subsystem:

```bash
python -m cli.aisp bench run --targets ch20:integrated_kv_cache --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch20:bf16_mlp --profile deep_dive --single-gpu
```

That is the right place to answer whether the gain came from overlap, memory movement, or simply removing one obvious bottleneck from the baseline.

## Repro Commands
```bash
python -m ch20.compare
python -m cli.aisp bench list-targets --chapter ch20
python -m cli.aisp bench run --targets ch20 --profile minimal
python -m cli.aisp tools ch20-ai-kernel-generator -- --device cpu --seqlen 512
python -m cli.aisp tools ch20-ai-kernel-workflow -- --device cpu --seqlen 512
```

## Learning Goals
- Chain memory, pipeline, and KV-cache optimizations together to see cumulative impact.
- Generate automatic reports that compare baseline vs tuned end-to-end runs.
- Prototype new kernels via the AI kernel generator and slot them into the harness.
- Validate improvements with workload-specific acceptance tests.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_bf16_mlp.py`, `optimized_bf16_mlp.py`, `ai_kernel_generator.py`, `core/optimization/inductor_guard.py` | Precision-policy workload plus the shared Inductor cudagraph guard used by the compiled end-to-end paths. |
| `baseline_pipeline_sequential.py`, `optimized_pipeline_sequential.py`, `baseline_end_to_end_bandwidth.py`, `optimized_end_to_end_bandwidth.py` | Pipeline and bandwidth case studies showing how optimizations interact across stages. `pipeline_sequential` currently remains informational after the fairness refresh. |
| `baseline_integrated_kv_cache.py`, `optimized_integrated_kv_cache.py` | Integrated KV-cache demos contrasting naive per-token cache handling with the restored block-wise paged-cache path used by the end-to-end benchmark. |
| `baseline_memory_standard.py`, `optimized_memory_standard.py` | Memory-focused harness verifying allocator changes at system level. |
| `baseline_training_single.py`, `optimized_training_single.py`, `test.cu`, `Makefile` | Single-device training case study plus CUDA kernels used in the final report. |
| `compare.py`, `arch_config.py`, `expectations_{hardware_key}.json` | Harness driver, architecture settings, and expectation baselines. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch20.compare
python -m cli.aisp bench list-targets --chapter ch20
python -m cli.aisp bench run --targets ch20 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch20.compare` emits per-stage summaries that show each optimized variant meeting or exceeding stored expectations.
- `python -m ch20.ai_kernel_generator --emit test.cu` produces CUDA kernels that compile via `nvcc` and integrate into the harness without manual edits.
- `python -m cli.aisp bench run --targets ch20:integrated_kv_cache --profile deep_dive` is the stronger canonical end-to-end paged-cache proof after restoring block-wise processing in the optimized path.

## Notes
- `core/optimization/inductor_guard.py` is the canonical helper for gating Inductor cudagraph features in the compiled chapter 20 paths.
- `ai_kernel_generator.py` logs generated code to `artifacts/` for reproducibility; capture the log with your proof-of-benefit bundle.
