# Lab - MoE CUDA PTX

## Summary
Benchmarks a routed top-2 SwiGLU MoE FFN with a conservative BF16 reference path and a staged optimized CUDA path. The lab is built as a standalone MoE kernel story: routing, token packing, grouped expert compute, MXFP8-style quantization surfaces, and end-to-end layer timing all live here.

## Problem
MoE optimization claims usually blur together three different costs:

- routing and token packing,
- grouped expert GEMM, and
- quantization / layout preparation for low-precision kernels.

This lab keeps those surfaces separate while still providing an end-to-end single-layer benchmark.

## Baseline Path
- BF16 reference path
- explicit top-2 token-to-expert dispatch
- per-expert Python / eager grouped execution
- conservative quantization path with explicit reshape / transpose work

## Optimized Path
- grouped expert execution on pre-packed token buckets
- vectorized expert BMM path for forward and backward
- MXFP8-style activation quantization surface benchmarked separately via `moe_quant`
- end-to-end forward path that keeps routing visible in the timed region without a standalone quantize/dequantize round trip

## Targets
| Target | Path |
| --- | --- |
| `labs/moe_cuda_ptx:moe_quant` | BF16 -> MXFP8-style quantization surface |
| `labs/moe_cuda_ptx:moe_grouped_gemm_fwd` | Grouped expert forward FFN surface |
| `labs/moe_cuda_ptx:moe_grouped_gemm_bwd` | Grouped expert forward+backward FFN surface |
| `labs/moe_cuda_ptx:moe_layer` | End-to-end routed top-2 SwiGLU MoE layer |

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_cuda_ptx
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_layer --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_grouped_gemm_fwd --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_grouped_gemm_bwd --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_quant --profile minimal
```

Useful debug overrides:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_layer \
  --target-extra-arg labs/moe_cuda_ptx:moe_layer="--num-tokens 4096 --hidden-dim 2048 --expert-ffn-dim 1024 --mode forward --histogram skewed"
```

Backward verification shape:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_grouped_gemm_bwd \
  --target-extra-arg labs/moe_cuda_ptx:moe_grouped_gemm_bwd="--num-tokens 4096 --hidden-dim 2048 --expert-ffn-dim 1024 --mode fwd_bwd"
```

## Current Status
- `moe_quant` is strict-harness verified with `--profile minimal` on a debug shape and currently shows a directional CUDA win.
- `moe_grouped_gemm_fwd` is strict-harness verified with `--profile minimal` on a debug shape and currently shows a directional CUDA win after moving the microbenchmark to the packed grouped-kernel core.
- `moe_grouped_gemm_bwd` and `moe_layer` are implemented and direct-runtime verified, but the current strict harness intermittently misclassifies transient `/usr/bin/python` GPU processes as foreign during some runs on this host.
- The PTX path is still an explicit Blackwell-gated scaffold. The current optimized backend is CUDA.

Debug-shape verification used during implementation:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_quant \
  --profile minimal \
  --target-extra-arg 'labs/moe_cuda_ptx:moe_quant=--num-tokens 2048 --hidden-dim 1024 --expert-ffn-dim 512 --mode forward'

python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_grouped_gemm_fwd \
  --profile minimal \
  --target-extra-arg 'labs/moe_cuda_ptx:moe_grouped_gemm_fwd=--num-tokens 2048 --hidden-dim 1024 --expert-ffn-dim 512 --mode forward'
```

## Learning Goals
- Keep routing local to the lab instead of hiding it behind another chapter.
- Show why grouped expert execution wins over eager per-expert masking.
- Measure quantization work as a first-class MoE cost, not a hidden helper stage.
- Leave a clear upgrade path for a future Blackwell `tcgen05` PTX backend.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_moe_*.py`, `optimized_moe_*.py` | Thin benchmark wrappers discovered by the harness. |
| `moe_cuda_ptx_common.py` | Shared workload config, routing, grouped FFN paths, quantization, and benchmark class. |
| `moe_cuda_ptx_extension.py`, `moe_cuda_ptx_stub.cu` | PTX-backend scaffold and Blackwell gating for future milestone work. |
| `expectations_{hardware_key}.json` | Reserved for strict expectations once the lab settles. |

## Notes
- The current optimized path is the staged CUDA milestone, not the final PTX/tcgen05 backend.
- The PTX scaffold is intentionally explicit and Blackwell-gated so the future tcgen05 path can land without changing the benchmark interface.
