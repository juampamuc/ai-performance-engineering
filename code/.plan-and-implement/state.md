## 2026-03-20T14:39:00Z

- Objective: Review all chapter and lab performance optimizations so every comparison is valid, fair, reproducible, and clearly demonstrates the optimization intent described in `book-after/ch*.md` for chapter code and in lab READMEs for lab examples.
- Acceptance criteria:
- Every reviewable `chXX/` benchmark/example pair is checked against the chapter manuscript intent.
- Every reviewable `labs/` benchmark-pair lab is checked against its README and repo lab-quality bar.
- Invalid benchmark comparisons are fixed locally rather than documented away.
- Non-pair workflow or matrix labs are labeled and documented honestly instead of pretending to be strict baseline/optimized comparisons.
- Each touched runtime path has explicit verification evidence: syntax/import validation plus at least one realistic repo invocation when feasible.
- Constraints:
- Keep existing files and user changes as-is; do not revert or delete files.
- Follow the repo benchmark methodology: frozen workload, one variable at a time, multiple trials where claims matter, and visible provenance.
- Keep execution local in this thread; no delegated sub-agent loop was requested.
- Current status: inventory and audit-tooling bootstrap completed. The benchmark-pair audit workflow is now doc-aware, and the next active phase is running chapter batches against the improved workflow.
