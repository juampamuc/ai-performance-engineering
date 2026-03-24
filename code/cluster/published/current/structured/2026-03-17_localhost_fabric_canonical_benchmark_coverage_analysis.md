# Benchmark Coverage Analysis: `2026-03-17_localhost_fabric_canonical`

Generated: `2026-03-17T03:00:15.133911+00:00`

| Field | Value |
|---|---|
| Labels | `localhost` |
| Coverage score | `100%` |
| Maturity | `high` |
| Advanced coverage score | `82%` |

## Subsystem Coverage

| Subsystem | Covered |
|---|---|
| `sm_compute` | `yes` |
| `hbm_memory` | `yes` |
| `gpu_gpu_communication` | `yes` |
| `gpu_cpu_transfer` | `yes` |
| `ai_workloads` | `yes` |

Missing: `none`

## Advanced Coverage

| Advanced signal | Covered |
|---|---|
| `vllm_request_rate_sweep` | `no` |
| `vllm_concurrency_repeat_stability` | `yes` |
| `vllm_request_rate_repeat_stability` | `no` |
| `fio_repeat_stability` | `yes` |
| `allreduce_stability` | `yes` |
| `allreduce_latency_comp` | `yes` |
| `allgather_control_plane` | `yes` |
| `nccl_alltoall` | `yes` |
| `nccl_algo_comparison` | `yes` |
| `train_step_workload` | `yes` |
| `mlperf_alignment` | `yes` |

## Recommended Next Runs

- Coverage is complete across the five major subsystems.
