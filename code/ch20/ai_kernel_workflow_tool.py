"""Chapter 20 AI-assisted kernel workflow tool.

Runs a small end-to-end workflow that:
1. evaluates the Chapter 20 FlexAttention kernel candidate,
2. summarizes manual verification checks, and
3. summarizes ProofWright-style formal verification checks.

The goal is not to simulate a full RL loop, but to give the chapter a single
supported entrypoint that ties generation and verification together.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from ch20 import ai_kernel_generator
from ch20.kernel_verification_tool import ManualKernelVerifier
from ch20.proofwright_verify_tool import ProofWrightAgent


def _gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


def _reference_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


KERNEL_SOURCE = """
__global__ void gelu_kernel(float* input, float* output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        float x = input[tid];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[tid] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
""".strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 20 AI-assisted kernel workflow tool")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--output-json", type=Path, help="Optional path to write the workflow summary")
    return parser.parse_args()


def _run_kernel_candidate(args: argparse.Namespace) -> Dict[str, Any]:
    ai_kernel_generator._ensure_environment(args.device)
    dtype = ai_kernel_generator._resolve_dtype(args.dtype)
    seqlen = args.seqlen
    if args.device == "cpu":
        if dtype is not torch.float32:
            dtype = torch.float32
        seqlen = min(seqlen, 2048)

    start = time.perf_counter()
    max_err = ai_kernel_generator.run_once(
        batch=args.batch,
        heads=args.heads,
        seqlen=seqlen,
        head_dim=args.head_dim,
        dtype=dtype,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "device": args.device,
        "dtype": str(dtype).replace("torch.", ""),
        "batch": args.batch,
        "heads": args.heads,
        "seqlen": seqlen,
        "head_dim": args.head_dim,
        "max_abs_error": float(max_err),
        "single_run_ms": elapsed_ms,
    }


def _run_manual_verification(device: str) -> Dict[str, Any]:
    verifier = ManualKernelVerifier(device=device)
    shape = (128, 128)
    random_pass, random_errors = verifier.random_test(_gelu_kernel, _reference_gelu, shape=shape, num_tests=5)
    edge_pass, edge_errors = verifier.edge_case_test(_gelu_kernel, _reference_gelu, shape=shape)
    boundary_pass, boundary_errors = verifier.boundary_test(_gelu_kernel, _reference_gelu, base_shape=shape)
    return {
        "random_tests": {"passed": random_pass, "error_count": len(random_errors)},
        "edge_cases": {"passed": edge_pass, "error_count": len(edge_errors)},
        "boundary_tests": {"passed": boundary_pass, "error_count": len(boundary_errors)},
        "formal_proof": False,
    }


def _run_proofwright_verification(device: str) -> Dict[str, Any]:
    agent = ProofWrightAgent(device=device)
    spec = agent._generate_spec_from_kernel(KERNEL_SOURCE)
    memory = agent.verify_memory_safety(KERNEL_SOURCE, spec)
    thread = agent.verify_thread_safety(KERNEL_SOURCE, spec)
    semantic = agent.verify_semantic_correctness(
        _gelu_kernel,
        _reference_gelu,
        input_shapes=[(128, 128), (64, 64)],
        device=device,
    )
    report = agent.generate_verification_report()
    return {
        "memory_safety": memory.to_dict(),
        "thread_safety": thread.to_dict(),
        "semantic_correctness": semantic.to_dict(),
        "summary": report.get("summary", {}),
        "formal_proof": True,
    }


def main() -> int:
    args = _parse_args()
    workflow = {
        "kernel_candidate": _run_kernel_candidate(args),
        "manual_verification": _run_manual_verification(args.device),
        "proofwright_verification": _run_proofwright_verification(args.device),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")
    print(json.dumps(workflow, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
