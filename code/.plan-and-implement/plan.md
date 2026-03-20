## 2026-03-20T14:39:00Z

1. Inventory the audit surface.
   Confirm chapter coverage, lab coverage, manuscript/README sources of truth, and current benchmark methodology constraints.
2. Build the audit rubric and lightweight tooling.
   Capture the checks that distinguish a valid comparison from a misleading one: frozen workload, isolated variable, correctness parity, honest naming/doc shape, reproducible harness path, and measurable artifact coverage.
3. Audit chapters batch by batch.
   Start with chapters that have either existing remediation packets or active worktree changes, then continue through the remaining chapters.
   First target batch: `ch10`, `ch11`, `ch14`, `ch16`, `ch18`, followed by the currently touched `ch01`, `ch02`, `ch09`, `ch12`, `ch13`, `ch20`.
4. Audit labs batch by batch.
   Separate benchmark-pair labs from playbook/matrix labs and fix code/docs/harness mismatches instead of forcing fake pair semantics.
   Use the new `scope_contract` audit step to decide whether each lab should be treated as a strict pair review or an honest workflow/playbook review.
5. Verify every touched path and maintain an evidence ledger.
   Record exact commands, outcomes, and any remaining blockers or non-feasible runs.
