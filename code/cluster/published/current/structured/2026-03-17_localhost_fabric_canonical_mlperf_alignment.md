# MLPerf Alignment: `2026-03-17_localhost_fabric_canonical`

Generated: `2026-03-17T03:00:15.055723+00:00`

| Field | Value |
|---|---|
| Overall status | `training_ready_only` |
| Inference track ready | `False` |
| Training track ready | `True` |

## Inference Track

| Signal | Ready |
|---|---|
| `concurrency_sweep` | `yes` |
| `request_rate_sweep` | `no` |
| `concurrency_slo_goodput` | `yes` |
| `request_rate_slo_goodput` | `no` |
| `concurrency_repeat_stability` | `yes` |
| `request_rate_repeat_stability` | `no` |

## Training Track

| Signal | Ready |
|---|---|
| `train_step_workload` | `yes` |
| `nccl_collectives_single_or_multi` | `yes` |
| `allreduce_stability` | `yes` |
| `alltoall_moe_coverage` | `yes` |
| `multinode_train_step` | `no` |

## References

- Inference: `MLPerf Inference Datacenter (LLM-style serving: throughput + TTFT/TPOT tails)`
- Training: `MLPerf Training (LLM-style train-step + distributed collective behavior)`
- Future-facing LLM set: `llama3.1_8b, llama3.1_405b, gpt_oss_20b`

## Recommendations

- Complete inference track: concurrency + request-rate sweeps with SLO goodput and repeat stability artifacts.
- Add multinode train-step evidence for distributed training alignment.
