# Canonical 2-Node Inference Surface

This is the current frozen serving lane for the core 2-node inference workflow.

It does **not** introduce a new execution backend. It packages the existing repo surfaces into one repeatable contract that another engineer can run immediately.

## Contract Summary

| Contract area | Canonical value |
| --- | --- |
| Execution backend | `python -m cli.aisp cluster common-eval` |
| Dry-run / preflight gate | `--preset multinode-readiness` |
| Full benchmark bundle | `--preset modern-llm` |
| Serving stack | vLLM Ray-based multi-node serving |
| Model lane | `openai/gpt-oss-120b` |
| Precision | `bf16` |
| Request shape | `ISL=1024`, `OSL=1024` |
| Default concurrency point | `64` |
| Runtime image family | `vllm/vllm-openai:cu130-nightly` (`-aarch64` on arm64 hosts) |
| Runtime image rule for canonical runs | resolve once, pin to `repo@sha256:...`, and reuse the same digest for baseline and candidate |
| Topology assumption | exactly 2 hosts, homogeneous visible GPU shape, tensor parallel resolved across all visible GPUs on both nodes |
| Interface requirement | explicit `--oob-if` and `--socket-ifname` for 2-node runs |
| Artifact root | `cluster/runs/<run_id>/` |

## Why This Surface

- `multinode-readiness` already provides a no-workload dry-run that validates the exact 2-node contract and writes a structured artifact plus manifest.
- `modern-llm` already provides the high-signal cluster bundle, and in 2-node scope it includes multinode vLLM plus the communication artifacts needed for bottleneck analysis.
- The multinode launcher already defaults to the repo's current serving lane: `openai/gpt-oss-120b` on `vllm/vllm-openai:cu130-nightly` with tensor parallel across both nodes.

## Step 1: Dry-Run / Preflight

This command validates the 2-node contract and writes the readiness artifact and manifest **without launching workloads**.

```bash
python -m cli.aisp cluster common-eval \
  --preset multinode-readiness \
  --run-id <run_id> \
  --hosts <leader,worker> \
  --labels <leader_label,worker_label> \
  --ssh-user <user> \
  --oob-if <iface> \
  --socket-ifname <iface>
```

Expected outputs:

- `cluster/runs/<run_id>/structured/<run_id>_multinode_readiness.json`
- `cluster/runs/<run_id>/manifest.json`

Important behavior:

- the readiness command exits before benchmark execution
- it validates host count, label count, duplicate labels, interface bindings, and multi-node contract toggles
- it is safe to use as a dry-run even before worker credentials or benchmark images are fully ready

## Step 2: Canonical 2-Node Run

After readiness is green, run the canonical modern-LLM bundle on exactly 2 hosts.

```bash
python -m cli.aisp cluster common-eval \
  --preset modern-llm \
  --run-id <run_id> \
  --hosts <leader,worker> \
  --labels <leader_label,worker_label> \
  --ssh-user <user> \
  --ssh-key <path> \
  --oob-if <iface> \
  --socket-ifname <iface> \
  --primary-label <leader_label> \
  --extra-arg --gpu-hourly-cost-usd \
  --extra-arg <usd_per_gpu_hour> \
  --extra-arg --vllm-multinode-image \
  --extra-arg <repo@sha256:...>
```

Canonical-run rules:

- pass the same `--vllm-multinode-image` digest for baseline and candidate runs
- keep `HOSTS`, `LABELS`, interface bindings, model, precision, ISL, OSL, and TP behavior unchanged across paired runs
- record cost assumptions with `--gpu-hourly-cost-usd` and keep that value constant across comparisons
- use a fresh `run_id` per replicate; for publication-grade comparisons, collect at least 3 replicates per side

## Minimum Evidence Bundle

Treat these as the minimum bundle required before handing the run to optimization or reporting workflows.

| Evidence family | Required artifacts |
| --- | --- |
| Run lineage | `cluster/runs/<run_id>/manifest.json`, `cluster/runs/<run_id>/structured/<run_id>_suite_steps.json` |
| Preflight contract | `cluster/runs/<run_id>/structured/<run_id>_multinode_readiness.json`, `cluster/runs/<run_id>/structured/<run_id>_preflight_services.json` |
| Host + GPU inventory | `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_meta.json`, `cluster/runs/<run_id>/structured/<run_id>_<worker_label>_meta.json` |
| Serving result | `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_serve.json`, `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_serve.csv`, `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_serve.jsonl` |
| SLO goodput | `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_slo_goodput.json`, `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_slo_goodput.csv` |
| Clock-lock proof | `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_vllm_multinode_leader_clock_lock.json`, `cluster/runs/<run_id>/structured/<run_id>_<worker_label>_vllm_multinode_worker_clock_lock.json` |
| Communication | `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_nccl.json`, `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_allreduce_stability.json`, `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_allreduce_latency_comp.json`, `cluster/runs/<run_id>/structured/<run_id>_<leader_label>_allgather_control_plane.json` |
| Scorecard + coverage | `cluster/runs/<run_id>/structured/<run_id>_cluster_scorecard.json`, `cluster/runs/<run_id>/structured/<run_id>_benchmark_coverage_analysis.json`, `cluster/runs/<run_id>/structured/<run_id>_mlperf_alignment.json` |

## Telemetry Mapping

| Need | Source |
| --- | --- |
| GPU state and topology | `*_meta.json`, `*_vllm_multinode_*_clock_lock.json` |
| Host/runtime readiness | `*_preflight_services.json`, `*_multinode_readiness.json` |
| Network and collective behavior | `*_nccl.json`, `*_allreduce_stability.json`, `*_allreduce_latency_comp.json`, `*_allgather_control_plane.json` |
| Serving latency/throughput | `*_vllm_multinode_serve.json`, `*_vllm_multinode_serve.csv`, `*_vllm_multinode_slo_goodput.json` |
| Spend/cost assumptions | CLI flag `--gpu-hourly-cost-usd`, plus the paired workload/intake contract in `templates/canonical_2node_inference_benchmark_run.yaml` and `templates/canonical_2node_inference_workload_spec.yaml` |
| Final artifact inventory | `manifest.json` |

## Companion Contract Files

- Workload spec example: `templates/canonical_2node_inference_workload_spec.yaml`
- BenchmarkRun example: `templates/canonical_2node_inference_benchmark_run.yaml`
- Generic intake template for budget and KPI capture: `templates/performance_intake.yaml`

## Handoff Note For FLU-122

Use this handoff bundle when unblocking downstream optimization work:

1. run readiness-only first and attach `manifest.json` + `*_multinode_readiness.json`
2. run the canonical `modern-llm` bundle with a pinned `--vllm-multinode-image`
3. compare only runs that share the same host shape, interface bindings, model, request shape, and hourly cost assumption
4. treat the manifest plus the evidence bundle above as the minimum review packet

For `FLU-122`, keep the **same 2-node launcher, artifact root, telemetry bundle, image-pin rule, and cost metadata rule**, but override the model lane to the requested Mixtral-class workload. The model may change; the evidence contract should not.
