# Cluster Scorecard: `2026-03-17_localhost_fabric_canonical`

Generated: `2026-03-17T03:00:14.975882+00:00`
Workload KPI label: `localhost`

## Canonical Completeness

| Field | Value |
|---|---|
| Overall score | `0.0` |
| Pass/fail | `fail` |
| Coverage score | `n/a%` |
| Advanced coverage score | `n/a%` |
| Coverage maturity | `n/a` |
| MLPerf overall status | `n/a` |
| MLPerf inference track ready | `False` |
| MLPerf training track ready | `False` |
| Gate: coverage >= min | `False` |
| Gate: advanced >= min | `False` |
| Gate: MLPerf alignment minimum | `False` |
| Gate: canonical complete | `False` |

## Unified KPIs

| Domain | KPI | Value |
|---|---|---:|
| Compute | GEMM max TFLOPS | `1272.8` |
| Memory | nvbandwidth HBM GB/s | `0.0` |
| Memory | STREAM-like triad GB/s | `6191.7` |
| Communication | NCCL single-node peak busbw GB/s | `0.0` |
| Communication | NCCL multi-node peak busbw GB/s | `0.0` |
| Communication | Multi/single busbw ratio | `0.00` |
| Communication | NCCL all-to-all single-node peak busbw GB/s | `0.0` |
| Communication | NCCL all-to-all multi-node peak busbw GB/s | `0.0` |
| Communication | NCCL all-to-all multi/single busbw ratio | `0.00` |
| Communication | NCCL algo winner | `n/a (single-rank)` |
| Communication | NCCL algo spread % | `n/a` |
| Communication | NCCL auto gap % | `n/a` |
| Communication | Allreduce stability CV % | `n/a` |
| Communication | Allreduce stability p99/p50 | `n/a` |
| Communication | Allreduce jitter assessment | `n/a (world_size<=1)` |
| Communication | Allreduce latency comp (small/large duration ratio) | `237.24` |
| Communication | Allreduce latency comp one-large duration ms | `0.0388` |
| Communication | Allreduce latency comp many-small duration ms | `9.2050` |
| Communication | all_gather_object vs tensor speedup | `6.35x` |
| Communication | all_gather_object vs all_reduce speedup | `14.63x` |
| Communication | Control-plane fastest method | `all_reduce_tensor` |
| Communication | Control-plane fastest latency ms | `0.0251` |
| Fabric | Fabric scorecard status | `ok` |
| Fabric | Fabric completeness | `full_stack_verified` |
| Fabric | Fabric runtime/full-stack families | `1` / `1` |
| Fabric | Fabric management planes configured | `1` |
| Host transfer | nvbandwidth H2D GB/s | `55.6` |
| Workload | vLLM throughput gain ratio | `1.00` |
| Workload | vLLM p99 TTFT ratio | `1.00` |
| Workload | vLLM max SLO goodput tok/s | `484.49` |
| Workload | vLLM goodput efficiency ratio | `1.00` |
| Workload | vLLM knee concurrency | `0` |
| Workload | vLLM request-rate max tok/s | `0.00` |
| Workload | vLLM request-rate at max tok/s | `0.00` |
| Efficiency | vLLM tok/J @ max tok/s | `3.070` |
| Efficiency | vLLM request-rate tok/J @ max tok/s | `0.000` |
| Efficiency | Cost USD / 1M tok (concurrency) | `n/a` |
| Efficiency | Cost USD / 1M tok (request-rate) | `n/a` |
| Workload Stability | vLLM conc tok/s CV p95 % | `0.00` |
| Workload Stability | vLLM conc p99 TTFT CV p95 % | `0.00` |
| Workload Stability | vLLM rate tok/s CV p95 % | `n/a` |
| Storage Stability | fio seq-read BW CV % | `2.72` |
| Storage Stability | fio seq-write BW CV % | `2.29` |

## Fabric Summary

| Field | Value |
|---|---|
| Fabric status | `ok` |
| Fabric completeness | `full_stack_verified` |
| Management planes configured | `1` |
| Runtime-verified families | `1` |
| Full-stack-verified families | `1` |
| Families present | `nvlink` |
| Full-stack families | `nvlink` |

## Bottleneck Classification

| Classifier | Value |
|---|---|
| Dominant bottleneck | `mixed` |
| Confidence | `low` |

| Evidence |
|---|
| No single dominant subsystem bottleneck exceeded heuristic trigger thresholds. |

| Recommended next actions |
|---|
| Treat this run as mixed-bound; prioritize top workload hotspots. |
| Run targeted deep-dive profiling on the slowest end-to-end workloads. |

## Per-Node Metrics

| Label | GEMM max TFLOPS | nvbandwidth HBM GB/s | STREAM triad GB/s | vLLM tok/s gain | vLLM p99 TTFT ratio | vLLM max SLO goodput tok/s | vLLM knee concurrency | vLLM conc tok/s CV p95 % | fio seq read MB/s | fio seq read CV % | fio seq write MB/s | fio seq write CV % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `localhost` | `1272.8` | `0.0` | `6191.7` | `1.00` | `1.00` | `484.49` | `0` | `0.00` | `1136.6` | `2.72` | `698.8` | `2.29` |
