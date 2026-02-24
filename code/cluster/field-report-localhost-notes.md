# Cluster Case Study Field Notes (Localhost Package)

Last updated: 2026-02-24. Canonical run: `2026-02-24_localhost_fullsuite_r4`.

## Table of Contents
1. [Scope](#scope)
2. [Required Reliability Gates](#required-reliability-gates)
3. [Operator Friction + Monitoring](#operator-friction--monitoring)
4. [Required Issue Ledger](#required-issue-ledger)
5. [Root Cause + Fix Mapping](#root-cause--fix-mapping)
6. [Evidence Matrix](#evidence-matrix)
7. [Repro Entry Point](#repro-entry-point)

## Scope
| Item | Value |
| --- | --- |
| Host | `localhost` |
| GPU count | `1` |
| Canonical run | `2026-02-24_localhost_fullsuite_r4` |
| Manifest | [results/structured/2026-02-24_localhost_fullsuite_r4_manifest.json](results/structured/2026-02-24_localhost_fullsuite_r4_manifest.json) |
| Suite steps | [results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json](results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json) |
| Operator dashboard | [results/structured/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.json](results/structured/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.json) |

## Required Reliability Gates
| Gate | Status | Evidence |
| --- | --- | --- |
| Hang triage readiness | `ok` | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_hang_triage_readiness.json](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_hang_triage_readiness.json) |
| Torchrun connectivity probe | `ok` | [results/structured/2026-02-24_localhost_fullsuite_r4_torchrun_connectivity_probe.json](results/structured/2026-02-24_localhost_fullsuite_r4_torchrun_connectivity_probe.json) |
| NCCL env sensitivity | `ok` | [results/structured/2026-02-24_localhost_fullsuite_r4_nccl_env_sensitivity.json](results/structured/2026-02-24_localhost_fullsuite_r4_nccl_env_sensitivity.json) |

## Operator Friction + Monitoring
| Check | Status | Evidence |
| --- | --- | --- |
| quick_friction | `degraded` | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_quick_friction.json](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_quick_friction.json) |
| monitoring_expectations | `ok` | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_monitoring_expectations.json](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_monitoring_expectations.json) |
| operator checks dashboard (json) | generated | [results/structured/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.json](results/structured/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.json) |
| operator checks dashboard (fig) | generated | [docs/figures/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.png](docs/figures/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.png) |

## Required Issue Ledger
| Required issue (verbatim) | Status in localhost package | Evidence |
| --- | --- | --- |
| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Not applicable (single-node scope) | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_fio.json](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_fio.json) |
| No multinode vLLM artifact in canonical package. | Not applicable (single-node scope) | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_vllm_serve_sweep.csv](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_vllm_serve_sweep.csv) |
| No nvbandwidth bundle in canonical package. | Intentionally skipped for this localhost package | [results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json](results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json) |
| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Not applicable (`health-suite off`) | [results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json](results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json) |
| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Not observed in localhost canary sweep (`c=1,2`) | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_vllm_serve_sweep.csv](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_vllm_serve_sweep.csv) |

## Root Cause + Fix Mapping
| Issue | Root cause | Fix | Verification |
| --- | --- | --- | --- |
| preflight false negatives | pipefail + unit-list pipeline fragility | use `systemctl show LoadState` in preflight | clean `r4` preflight (`rc=0`) |
| NVLink plot parse failure | parser too strict for single-GPU header format | robust tokenization + leading GPU header extraction | clean topology plot in `r4` |
| vLLM docker access failure | user session lacked docker group | add current user via `id -un` + `usermod -aG docker` | clean `r4` vLLM step (`rc=0`) |

## Evidence Matrix
| Claim | Evidence | Verdict |
| --- | --- | --- |
| Localhost suite is clean | [results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json](results/structured/2026-02-24_localhost_fullsuite_r4_suite_steps.json) | Backed |
| Operator checks now included | [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_quick_friction.json](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_quick_friction.json), [results/structured/2026-02-24_localhost_fullsuite_r4_localhost_monitoring_expectations.json](results/structured/2026-02-24_localhost_fullsuite_r4_localhost_monitoring_expectations.json), [results/structured/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.json](results/structured/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.json) | Backed |
| Report visuals are complete for local package | [docs/figures/2026-02-24_localhost_fullsuite_r4_cluster_story_dashboard.png](docs/figures/2026-02-24_localhost_fullsuite_r4_cluster_story_dashboard.png), [docs/figures/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.png](docs/figures/2026-02-24_localhost_fullsuite_r4_operator_checks_dashboard.png) | Backed |

## Repro Entry Point
| Step | Command |
| --- | --- |
| Re-run localhost canonical package | `sg docker -c "cluster/scripts/run_cluster_eval_suite.sh --run-id 2026-02-24_localhost_fullsuite_r4 --hosts localhost --labels localhost --ssh-user $(id -un) --primary-label localhost --skip-bootstrap-nodes --disable-fp4 --health-suite off --skip-vllm-multinode --model openai-community/gpt2 --tp 1 --isl 128 --osl 64 --concurrency-range '1 2' --port 8893 --fio-runtime 15 --skip-nvbandwidth"` |
