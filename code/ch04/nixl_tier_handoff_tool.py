"""Chapter 4 NIXL tier-handoff wrapper.

Reuses the labs/nccl_nixl_nvshmem tier-handoff runner so the Chapter 4
communication chapter has a direct, chapter-local entrypoint for the
book's NIXL-style memory-tier movement story.
"""

from __future__ import annotations

from labs.nccl_nixl_nvshmem.run_lab_nccl_nixl_nvshmem import main


if __name__ == "__main__":
    raise SystemExit(main())
