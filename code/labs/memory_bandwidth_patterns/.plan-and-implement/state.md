2026-03-20T04:09:00Z

Objective
- Build `labs/memory_bandwidth_patterns/` from the local GTC deck material as a self-contained lab with one honest benchmark pair, a runnable local milestone runner, and real B200 validation.

Acceptance Criteria
- Keep all source changes inside `labs/memory_bandwidth_patterns/`.
- Study the existing lab patterns first and match the repo’s pair-plus-runner shape.
- Cover measurable bandwidth patterns, not a generic summary.
- Include coalesced vs strided access, vectorized loads, shared-memory staging, and a cp.async-style double-buffered milestone when supported.
- Make unsupported advanced features fail fast instead of silently downgrading.
- Deliver README, runnable targets, and real validation evidence.

Status
- Implementation complete.
- Standalone runner validated on B200.
- Harness benchmark pair validated through `python -m cli.aisp bench run`.
- Remaining work limited to final summary and any follow-up callouts outside the owned directory.
