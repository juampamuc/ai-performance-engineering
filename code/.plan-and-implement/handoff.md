## 2026-03-20T14:39:00Z

- Decision: use the `plan-and-implement` shared-state workflow, but keep all planning and implementation in the main thread because the user did not authorize sub-agent delegation.
- Open question to resolve during the audit: which issues can be enforced mechanically with a repo audit script and which require manuscript-level manual review.
- Working assumption: chapter source of truth is `book-after/chN.md` or `book-after/chNN.md`; lab source of truth is the corresponding lab `README.md`, plus `labs/README.md` for pair-vs-playbook expectations.
- Resolved for tooling: the benchmark-pair audit now records a `scope_contract` for each reviewed scope, including source doc path, lab review mode (`benchmark-pair`, `benchmark-story`, `playbook-matrix`, or `challenge-kernel-lab`), and contract findings.
- New blocker removed: the nested audit pytest bundle was failing on a too-tight isolated-pair timeout test; the timeout now allows the descendant process to spawn before cleanup assertions run.
