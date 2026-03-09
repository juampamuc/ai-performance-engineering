# Lab - MoE Parallelism Planner

## Summary
Scenario planning tool for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules.

## Learning Goals
- Quantify memory budgets for experts, routers, and KV caches before deploying models.
- Explore different grouping strategies (hashing, topology-aware) and their throughput impact.
- Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.
- Simulate pipeline schedules to identify bottlenecks before touching production systems.

## Directory Layout
| Path | Description |
| --- | --- |
| `run_lab.py`, `scenarios.py`, `plan.py` | Tool entry point + canonical scenario definitions and sizing model. |
| `benchmarking.py` | Optional harness-compatible wrapper for ad-hoc integration; not currently a public baseline/optimized benchmark pair. |

## Running the Tool
Use the tool entrypoint or the direct script when you want reproducible scenario comparisons. This lab does not currently expose public baseline/optimized harness targets.
```bash
python -m cli.aisp tools moe-parallelism -- --scenario memory_budget
python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200
python labs/moe_parallelism/run_lab.py --scenario deepseek_gb200
```
- `python -m cli.aisp bench list-targets --chapter labs/moe_parallelism` intentionally returns no benchmark pairs today.
- If this lab is later promoted into harness targets, add explicit `baseline_*.py` and `optimized_*.py` entrypoints instead of implying them in the README.

## Validation Checklist
- `python -m cli.aisp tools moe-parallelism -- --scenario memory_budget` runs a single scenario via the tool registry.
- `python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200` runs a larger cluster scenario.
- `python labs/moe_parallelism/run_lab.py --scenario deepseek_gb200` runs the planner directly (without aisp).

## Notes
- Baseline vs optimized here are *planning* scenarios (different designs), not comparable performance benchmarks.
- `plan.py` centralizes scenario definitions so you only update one file when adding a new topology.
