#!/usr/bin/env python3
"""Audit and update `get_custom_metrics()` implementations.

This tool now treats metric truthfulness as the primary concern:
1. classify implementations as `real`, `helper-backed`, or `phantom`
2. reject placeholder timing/performance defaults that fabricate throughput
3. keep helper suggestions limited to chapters with coherent shared semantics

Usage:
    python core/scripts/update_custom_metrics.py --analyze
    python core/scripts/update_custom_metrics.py --validate
    python core/scripts/update_custom_metrics.py --show-suggestion ch07/baseline_memory_access.py
    python core/scripts/update_custom_metrics.py --apply --file ch07/baseline_memory_access.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CHAPTER_METRIC_HELPERS = {
    1: "compute_environment_metrics",
    2: "compute_memory_transfer_metrics",
    3: "compute_system_config_metrics",
    4: "compute_distributed_metrics",
    5: None,
    6: "compute_kernel_fundamentals_metrics",
    7: "compute_memory_access_metrics",
    8: "compute_optimization_metrics",
    9: "compute_roofline_metrics",
    10: None,
    11: "compute_stream_metrics",
    12: "compute_graph_metrics",
    13: "compute_precision_metrics",
    14: None,
    15: "compute_inference_metrics",
    16: "compute_inference_metrics",
    17: "compute_inference_metrics",
    18: None,
    19: "compute_precision_metrics",
    20: "compute_ai_optimization_metrics",
}

HELPER_SIGNATURES = {
    "compute_environment_metrics": {
        "import": "from core.benchmark.metrics import compute_environment_metrics",
        "params": ["gpu_count", "gpu_memory_gb"],
        "defaults": {
            "gpu_count": "torch.cuda.device_count() if torch.cuda.is_available() else 0",
            "gpu_memory_gb": "(torch.cuda.get_device_properties(0).total_memory / float(1024 ** 3)) if torch.cuda.is_available() else 0.0",
        },
    },
    "compute_memory_transfer_metrics": {
        "import": "from core.benchmark.metrics import compute_memory_transfer_metrics",
        "params": ["bytes_transferred", "elapsed_ms", "transfer_type"],
        "defaults": {
            "bytes_transferred": "float(getattr(self, 'N', getattr(self, 'num_elements', 0)) * 4)",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', None)",
            "transfer_type": '"hbm"',
        },
    },
    "compute_system_config_metrics": {
        "import": "from core.benchmark.metrics import compute_system_config_metrics",
        "params": ["numa_nodes", "cpu_cores"],
        "defaults": {
            "numa_nodes": "getattr(self, 'numa_nodes', 0)",
            "cpu_cores": "getattr(self, 'cpu_cores', 0)",
        },
    },
    "compute_distributed_metrics": {
        "import": "from core.benchmark.metrics import compute_distributed_metrics",
        "params": ["world_size", "bytes_transferred", "elapsed_ms", "collective_type"],
        "defaults": {
            "world_size": "getattr(self, 'world_size', 1)",
            "bytes_transferred": "float(getattr(self, 'bytes_transferred', getattr(self, '_bytes_transferred', 0.0)))",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', None)",
            "collective_type": '"allreduce"',
        },
    },
    "compute_kernel_fundamentals_metrics": {
        "import": "from core.benchmark.metrics import compute_kernel_fundamentals_metrics",
        "params": ["num_elements", "num_iterations"],
        "defaults": {
            "num_elements": "int(getattr(self, 'N', getattr(self, 'num_elements', 0)))",
            "num_iterations": "int(getattr(self, 'iterations', 1))",
        },
    },
    "compute_memory_access_metrics": {
        "import": "from core.benchmark.metrics import compute_memory_access_metrics",
        "params": ["bytes_requested", "bytes_actually_transferred", "num_transactions", "optimal_transactions"],
        "defaults": {
            "bytes_requested": "float(getattr(self, 'bytes_requested', getattr(self, 'N', 0) * 4))",
            "bytes_actually_transferred": "float(getattr(self, 'bytes_actually_transferred', getattr(self, 'N', 0) * 4))",
            "num_transactions": "int(getattr(self, 'num_transactions', max(1, getattr(self, 'N', 1) // 32)))",
            "optimal_transactions": "int(getattr(self, 'optimal_transactions', max(1, getattr(self, 'N', 1) // 32)))",
        },
    },
    "compute_optimization_metrics": {
        "import": "from core.benchmark.metrics import compute_optimization_metrics",
        "params": ["baseline_ms", "optimized_ms", "technique"],
        "defaults": {
            "baseline_ms": "getattr(self, '_baseline_ms', None)",
            "optimized_ms": "getattr(self, '_last_elapsed_ms', getattr(self, '_optimized_ms', None))",
            "technique": '"optimization"',
        },
    },
    "compute_roofline_metrics": {
        "import": "from core.benchmark.metrics import compute_roofline_metrics",
        "params": ["total_flops", "total_bytes", "elapsed_ms", "precision"],
        "defaults": {
            "total_flops": "float(getattr(self, 'total_flops', getattr(self, 'N', 0) * 2))",
            "total_bytes": "float(getattr(self, 'total_bytes', getattr(self, 'N', 0) * 4))",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', None)",
            "precision": '"fp16"',
        },
    },
    "compute_stream_metrics": {
        "import": "from core.benchmark.metrics import compute_stream_metrics",
        "params": ["sequential_time_ms", "overlapped_time_ms", "num_streams", "num_operations"],
        "defaults": {
            "sequential_time_ms": "getattr(self, '_sequential_ms', None)",
            "overlapped_time_ms": "getattr(self, '_last_elapsed_ms', getattr(self, '_overlapped_ms', None))",
            "num_streams": "int(getattr(self, 'num_streams', 1))",
            "num_operations": "int(getattr(self, 'num_operations', 1))",
        },
    },
    "compute_graph_metrics": {
        "import": "from core.benchmark.metrics import compute_graph_metrics",
        "params": ["baseline_launch_overhead_us", "graph_launch_overhead_us", "num_nodes", "num_iterations"],
        "defaults": {
            "baseline_launch_overhead_us": "getattr(self, '_baseline_launch_us', None)",
            "graph_launch_overhead_us": "getattr(self, '_graph_launch_us', None)",
            "num_nodes": "int(getattr(self, 'num_nodes', 0))",
            "num_iterations": "int(getattr(self, 'num_iterations', 0))",
        },
    },
    "compute_precision_metrics": {
        "import": "from core.benchmark.metrics import compute_precision_metrics",
        "params": ["fp32_time_ms", "reduced_precision_time_ms", "precision_type"],
        "defaults": {
            "fp32_time_ms": "getattr(self, '_fp32_ms', None)",
            "reduced_precision_time_ms": "getattr(self, '_last_elapsed_ms', getattr(self, '_reduced_ms', None))",
            "precision_type": '"fp8"',
        },
    },
    "compute_inference_metrics": {
        "import": "from core.benchmark.metrics import compute_inference_metrics",
        "params": ["ttft_ms", "tpot_ms", "total_tokens", "total_requests", "batch_size", "max_batch_size"],
        "defaults": {
            "ttft_ms": "getattr(self, '_ttft_ms', None)",
            "tpot_ms": "getattr(self, '_tpot_ms', None)",
            "total_tokens": "int(getattr(self, 'total_tokens', getattr(self, '_total_tokens', 0)))",
            "total_requests": "int(getattr(self, 'total_requests', getattr(self, '_total_requests', 0)))",
            "batch_size": "int(getattr(self, 'batch_size', 1))",
            "max_batch_size": "int(getattr(self, 'max_batch_size', getattr(self, 'batch_size', 1)))",
        },
    },
    "compute_speculative_decoding_metrics": {
        "import": "from core.benchmark.metrics import compute_speculative_decoding_metrics",
        "params": ["draft_tokens", "accepted_tokens", "draft_time_ms", "verify_time_ms", "num_rounds"],
        "defaults": {
            "draft_tokens": "int(getattr(self, 'draft_tokens', getattr(self, '_draft_tokens', 0)))",
            "accepted_tokens": "getattr(self, '_accepted_tokens', None)",
            "draft_time_ms": "getattr(self, '_draft_ms', None)",
            "verify_time_ms": "getattr(self, '_verify_ms', None)",
            "num_rounds": "getattr(self, '_num_rounds', None)",
        },
    },
    "compute_storage_io_metrics": {
        "import": "from core.benchmark.metrics import compute_storage_io_metrics",
        "params": ["bytes_read", "bytes_written", "read_time_ms", "write_time_ms"],
        "defaults": {
            "bytes_read": "float(getattr(self, 'bytes_read', getattr(self, '_bytes_read', 0.0)))",
            "bytes_written": "float(getattr(self, 'bytes_written', getattr(self, '_bytes_written', 0.0)))",
            "read_time_ms": "getattr(self, '_read_time_ms', None)",
            "write_time_ms": "getattr(self, '_write_time_ms', None)",
        },
    },
    "compute_pipeline_metrics": {
        "import": "from core.benchmark.metrics import compute_pipeline_metrics",
        "params": ["num_stages", "stage_times_ms"],
        "defaults": {
            "num_stages": "int(getattr(self, 'num_stages', 0))",
            "stage_times_ms": "list(getattr(self, 'stage_times_ms', getattr(self, '_stage_times_ms', [])))",
        },
    },
    "compute_triton_metrics": {
        "import": "from core.benchmark.metrics import compute_triton_metrics",
        "params": ["num_elements", "elapsed_ms", "block_size", "num_warps"],
        "defaults": {
            "num_elements": "int(getattr(self, 'N', getattr(self, 'num_elements', 0)))",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', None)",
            "block_size": "int(getattr(self, 'BLOCK_SIZE', getattr(self, 'block_size', 0)))",
            "num_warps": "int(getattr(self, 'num_warps', 4))",
        },
    },
    "compute_ai_optimization_metrics": {
        "import": "from core.benchmark.metrics import compute_ai_optimization_metrics",
        "params": ["original_time_ms", "ai_optimized_time_ms", "suggestions_applied", "suggestions_total"],
        "defaults": {
            "original_time_ms": "getattr(self, '_original_ms', None)",
            "ai_optimized_time_ms": "getattr(self, '_last_elapsed_ms', getattr(self, '_optimized_ms', None))",
            "suggestions_applied": "getattr(self, '_suggestions_applied', None)",
            "suggestions_total": "getattr(self, '_suggestions_total', None)",
        },
    },
    "compute_moe_metrics": {
        "import": "from core.benchmark.metrics import compute_moe_metrics",
        "params": ["num_experts", "active_experts", "tokens_per_expert", "routing_time_ms", "expert_compute_time_ms"],
        "defaults": {
            "num_experts": "int(getattr(self, 'num_experts', 0))",
            "active_experts": "int(getattr(self, 'active_experts', 0))",
            "tokens_per_expert": "list(getattr(self, 'tokens_per_expert', getattr(self, '_tokens_per_expert', [])))",
            "routing_time_ms": "getattr(self, '_routing_ms', None)",
            "expert_compute_time_ms": "getattr(self, '_expert_compute_ms', None)",
        },
    },
}

HELPER_MEASUREMENT_PARAMS = {
    "compute_memory_transfer_metrics": {"elapsed_ms"},
    "compute_distributed_metrics": {"elapsed_ms"},
    "compute_optimization_metrics": {"baseline_ms", "optimized_ms"},
    "compute_roofline_metrics": {"elapsed_ms"},
    "compute_stream_metrics": {"sequential_time_ms", "overlapped_time_ms"},
    "compute_graph_metrics": {"baseline_launch_overhead_us", "graph_launch_overhead_us"},
    "compute_precision_metrics": {"fp32_time_ms", "reduced_precision_time_ms"},
    "compute_inference_metrics": {"ttft_ms", "tpot_ms"},
    "compute_speculative_decoding_metrics": {"draft_tokens", "accepted_tokens", "draft_time_ms", "verify_time_ms", "num_rounds"},
    "compute_storage_io_metrics": {"read_time_ms", "write_time_ms"},
    "compute_triton_metrics": {"elapsed_ms"},
    "compute_ai_optimization_metrics": {"original_time_ms", "ai_optimized_time_ms", "suggestions_applied", "suggestions_total"},
    "compute_moe_metrics": {"routing_time_ms", "expert_compute_time_ms"},
}

PRIVATE_MEASURED_ATTRS = {
    "_last_elapsed_ms",
    "_baseline_ms",
    "_optimized_ms",
    "_sequential_ms",
    "_overlapped_ms",
    "_baseline_launch_us",
    "_graph_launch_us",
    "_fp32_ms",
    "_reduced_ms",
    "_ttft_ms",
    "_tpot_ms",
    "_draft_tokens",
    "_accepted_tokens",
    "_draft_ms",
    "_verify_ms",
    "_num_rounds",
    "_original_ms",
    "_suggestions_applied",
    "_suggestions_total",
    "_read_time_ms",
    "_write_time_ms",
    "_routing_ms",
    "_expert_compute_ms",
    "_total_tokens",
    "_total_requests",
}

FRAMEWORK_MEASURED_ATTRS = {
    "_last_elapsed_ms",
}

PERFORMANCE_KEY_SUFFIXES = (
    "speedup",
    "improvement_pct",
    "ttft_ms",
    "tpot_ms",
    "draft_time_ms",
    "verify_time_ms",
    "achieved_gbps",
    "algo_bandwidth_gbps",
    "efficiency_pct",
    "achieved_tflops",
    "elements_per_second",
    "baseline_ms",
    "optimized_ms",
    "fp32_ms",
    "reduced_ms",
    "sequential_ms",
    "overlapped_ms",
    "baseline_launch_us",
    "graph_launch_us",
    "total_gbps",
    "read_gbps",
    "write_gbps",
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_chapter_from_path(path: Path) -> Optional[int]:
    match = re.search(r"ch(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1))


def has_conditional_none_return(content: str) -> bool:
    match = re.search(
        r"def get_custom_metrics\s*\([^)]*\)[^:]*:\s*\n((?:[ \t]+.*\n)*)",
        content,
    )
    if not match:
        return False
    impl = match.group(1)
    has_conditional_none = bool(re.search(r"if\s+.*:\s*\n\s*return None", impl))
    has_dict_return = bool(re.search(r"return\s*(?:\{|\w+\()", impl))
    return has_conditional_none and has_dict_return


def _read_text(path: Path) -> str:
    return path.read_text()


def _is_benchmark_candidate(path: Path, content: str) -> bool:
    if path.name == "__init__.py" or "tests" in path.parts:
        return False
    if path.name.startswith(("baseline_", "optimized_")):
        return True
    return (
        "def get_custom_metrics" in content
        or "def get_benchmark(" in content
        or "BaseBenchmark" in content
    )


def is_standalone_script(content: str) -> bool:
    has_benchmark_class = bool(
        re.search(r"class\s+\w+.*\((?:[^)]*BaseBenchmark|[^)]*BenchmarkBase|[^)]*Benchmark)\)", content)
    )
    has_main_func = "def main(" in content
    has_main_guard = "__name__" in content and "__main__" in content
    get_benchmark_match = re.search(r"def get_benchmark\(\)[^:]*:\s*\n((?:[ \t]+.*\n)*)", content)
    if not get_benchmark_match:
        return False
    body = get_benchmark_match.group(1)
    returns_callable = any(token in body for token in ("return main", "lambda", "def _run"))
    return (not has_benchmark_class) and has_main_func and has_main_guard and returns_callable


def is_alias_file(content: str) -> bool:
    if re.search(r"^class\s+\w+", content, re.MULTILINE):
        return False
    imports_benchmark = bool(
        re.search(r"from\s+[\w.]+\s+import\s+.*(?:Benchmark|get_benchmark)", content, re.DOTALL)
    )
    get_benchmark_match = re.search(r"def get_benchmark\(\)[^:]*:\s*\n((?:[ \t]+.*\n)*)", content)
    if not imports_benchmark or not get_benchmark_match:
        return False
    body = get_benchmark_match.group(1)
    return "return _get_benchmark()" in body or bool(re.search(r"return\s+\w+Benchmark\(\)", body))


def find_parent_class_file(parent_class: str, current_file: Path, root: Path) -> Optional[Path]:
    try:
        current_content = _read_text(current_file)
    except OSError:
        return None
    import_patterns = [
        rf"from\s+([\w.]+)\s+import\s+.*{re.escape(parent_class)}",
        rf"from\s+([\w.]+)\s+import\s+\(.*{re.escape(parent_class)}.*\)",
    ]
    for pattern in import_patterns:
        match = re.search(pattern, current_content, re.DOTALL)
        if not match:
            continue
        module_path = match.group(1).replace(".", "/")
        candidate = root / f"{module_path}.py"
        if candidate.exists():
            return candidate
    for py_file in current_file.parent.glob("*.py"):
        if py_file == current_file:
            continue
        try:
            content = _read_text(py_file)
        except OSError:
            continue
        if f"class {parent_class}" in content:
            return py_file
    return None


def check_class_has_get_custom_metrics(file_path: Path, class_name: str) -> bool:
    try:
        content = _read_text(file_path)
    except OSError:
        return False
    class_pattern = rf"class\s+{re.escape(class_name)}\s*\([^)]*\):\s*\n((?:[ \t]+.*\n)*)"
    match = re.search(class_pattern, content)
    if not match:
        return "def get_custom_metrics" in content
    return "def get_custom_metrics" in match.group(1)


def inherits_get_custom_metrics(content: str, file_path: Path, root: Path, visited: Optional[Set[Path]] = None) -> bool:
    visited = visited or set()
    if file_path in visited:
        return False
    visited.add(file_path)
    for parents in re.findall(r"class\s+\w+\s*\(([^)]+)\)", content):
        for parent in [part.strip().split(".")[-1] for part in parents.split(",")]:
            if parent == "BaseBenchmark":
                continue
            parent_file = find_parent_class_file(parent, file_path, root)
            if parent_file is None:
                continue
            if check_class_has_get_custom_metrics(parent_file, parent):
                return True
            try:
                parent_content = _read_text(parent_file)
            except OSError:
                continue
            if inherits_get_custom_metrics(parent_content, parent_file, root, visited):
                return True
    return False


def _get_get_custom_metrics_context(tree: ast.AST) -> Tuple[Optional[ast.FunctionDef], Optional[ast.ClassDef]]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "get_custom_metrics":
                    return child, node
        elif isinstance(node, ast.FunctionDef) and node.name == "get_custom_metrics":
            return node, None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_custom_metrics":
            return node, None
    return None, None


def _find_class_def(tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_self_private_getattr(node: ast.AST) -> Optional[Tuple[str, ast.AST]]:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
        return None
    if len(node.args) < 3:
        return None
    target, attr_node, default = node.args[:3]
    if not isinstance(target, ast.Name) or target.id != "self":
        return None
    if not isinstance(attr_node, ast.Constant) or not isinstance(attr_node.value, str):
        return None
    attr_name = attr_node.value
    if attr_name not in PRIVATE_MEASURED_ATTRS:
        return None
    return attr_name, default


def _is_literal_placeholder(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return False
        return isinstance(node.value, (int, float, str, bool))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal_placeholder(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is not None and not _is_literal_placeholder(key):
                return False
            if not _is_literal_placeholder(value):
                return False
        return True
    return False


def _collect_placeholder_reasons(node: ast.AST) -> List[str]:
    reasons: List[str] = []
    for child in ast.walk(node):
        match = _is_self_private_getattr(child)
        if match is None:
            continue
        attr_name, default = match
        if _is_literal_placeholder(default):
            reasons.append(f"{attr_name} uses a literal placeholder default")
    return reasons


def _collect_self_assigned_private_attrs(node: ast.AST) -> Set[str]:
    attrs: Set[str] = set()
    for child in ast.walk(node):
        targets: List[ast.AST] = []
        if isinstance(child, ast.Assign):
            targets.extend(child.targets)
        elif isinstance(child, ast.AnnAssign):
            targets.append(child.target)
        elif isinstance(child, ast.AugAssign):
            targets.append(child.target)

        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                attr_name = target.attr
                if attr_name.startswith("_"):
                    attrs.add(attr_name)

        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Name) or child.func.id != "setattr":
            continue
        if len(child.args) < 2:
            continue
        if not isinstance(child.args[0], ast.Name) or child.args[0].id != "self":
            continue
        if not isinstance(child.args[1], ast.Constant) or not isinstance(child.args[1].value, str):
            continue
        attr_name = child.args[1].value
        if attr_name.startswith("_"):
            attrs.add(attr_name)
    return attrs


def _base_class_names(class_node: ast.ClassDef) -> List[str]:
    names: List[str] = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _collect_class_hierarchy_private_attrs(
    file_path: Path,
    root: Path,
    class_node: Optional[ast.ClassDef],
    visited: Optional[Set[Tuple[Path, str]]] = None,
) -> Set[str]:
    if class_node is None:
        return set()

    visited = visited or set()
    visit_key = (file_path.resolve(), class_node.name)
    if visit_key in visited:
        return set()
    visited.add(visit_key)

    attrs = _collect_self_assigned_private_attrs(class_node)
    for parent_name in _base_class_names(class_node):
        if parent_name == "BaseBenchmark":
            continue
        parent_file = find_parent_class_file(parent_name, file_path, root)
        if parent_file is None:
            continue
        try:
            parent_content = _read_text(parent_file)
            parent_tree = ast.parse(parent_content)
        except (OSError, SyntaxError):
            continue
        parent_class = _find_class_def(parent_tree, parent_name)
        if parent_class is None:
            continue
        attrs.update(
            _collect_class_hierarchy_private_attrs(parent_file, root, parent_class, visited)
        )
    return attrs


def _collect_unassigned_private_attr_reasons(
    func: ast.FunctionDef,
    assigned_attrs: Set[str],
) -> List[str]:
    reasons: Set[str] = set()
    for child in ast.walk(func):
        match = _is_self_private_getattr(child)
        if match is None:
            continue
        attr_name, _default = match
        if attr_name in FRAMEWORK_MEASURED_ATTRS:
            continue
        if attr_name not in assigned_attrs:
            reasons.add(
                f"{attr_name} is read in get_custom_metrics but never assigned by the benchmark or its parents"
            )
    return sorted(reasons)


def _performance_literal_reasons(func: ast.FunctionDef) -> List[str]:
    reasons: List[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if not any(key.value.endswith(suffix) for suffix in PERFORMANCE_KEY_SUFFIXES):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                reasons.append(f"{key.value} is hard-coded to a literal performance value")
    return reasons


def _helper_placeholder_reasons(func: ast.FunctionDef) -> Tuple[Set[str], List[str]]:
    helper_names: Set[str] = set()
    reasons: List[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        helper_name = _call_name(node)
        if helper_name not in HELPER_SIGNATURES:
            continue
        helper_names.add(helper_name)
        measurement_params = HELPER_MEASUREMENT_PARAMS.get(helper_name, set())
        for kw in node.keywords:
            if kw.arg not in measurement_params:
                continue
            if _is_literal_placeholder(kw.value):
                reasons.append(f"{helper_name}.{kw.arg} uses a literal placeholder")
                continue
            reasons.extend(
                f"{helper_name}.{kw.arg}: {reason}"
                for reason in _collect_placeholder_reasons(kw.value)
            )
    return helper_names, reasons


def analyze_get_custom_metrics(file_path: Path, root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or get_repo_root()
    try:
        content = _read_text(file_path)
    except OSError as exc:
        return {"error": f"Could not read file: {exc}", "path": file_path}

    result: Dict[str, Any] = {
        "path": file_path,
        "has_method": False,
        "returns_none": False,
        "returns_empty": False,
        "returns_basic": False,
        "uses_helper": False,
        "helper_names": [],
        "has_conditional_none": False,
        "inherits_method": False,
        "is_standalone": is_standalone_script(content),
        "is_alias": is_alias_file(content),
        "line_number": None,
        "current_impl": None,
        "needs_update": False,
        "classification": "missing",
        "phantom_reasons": [],
    }
    if not _is_benchmark_candidate(file_path, content):
        result["classification"] = "skipped"
        return result

    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        result["error"] = f"SyntaxError: {exc}"
        result["classification"] = "error"
        return result

    func, owner_class = _get_get_custom_metrics_context(tree)
    if func is None:
        result["inherits_method"] = inherits_get_custom_metrics(content, file_path, root)
        if result["is_alias"]:
            result["classification"] = "alias"
        elif result["is_standalone"]:
            result["classification"] = "standalone"
        elif result["inherits_method"]:
            result["classification"] = "inherited"
        return result

    result["has_method"] = True
    result["line_number"] = func.lineno
    result["current_impl"] = ast.get_source_segment(content, func)
    result["has_conditional_none"] = has_conditional_none_return(content)

    assigned_attrs = _collect_class_hierarchy_private_attrs(file_path, root, owner_class)

    helper_names, helper_phantoms = _helper_placeholder_reasons(func)
    direct_phantoms = _collect_placeholder_reasons(func)
    unassigned_private_attr_reasons = _collect_unassigned_private_attr_reasons(func, assigned_attrs)
    perf_literal_reasons = _performance_literal_reasons(func)
    phantom_reasons = sorted(
        set(helper_phantoms + direct_phantoms + unassigned_private_attr_reasons + perf_literal_reasons)
    )
    result["helper_names"] = sorted(helper_names)
    result["uses_helper"] = bool(helper_names)
    result["phantom_reasons"] = phantom_reasons

    return_none = any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None
        for node in ast.walk(func)
    )
    return_empty = any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Dict) and not node.value.keys
        for node in ast.walk(func)
    )
    result["returns_none"] = return_none and not result["has_conditional_none"]
    result["returns_empty"] = return_empty

    if phantom_reasons:
        result["classification"] = "phantom"
        result["needs_update"] = True
    elif result["returns_empty"]:
        result["classification"] = "empty"
        result["needs_update"] = True
    elif result["returns_none"]:
        result["classification"] = "none"
        result["needs_update"] = True
    elif result["uses_helper"]:
        result["classification"] = "helper-backed"
    else:
        result["classification"] = "real"

    return result


def generate_helper_code(helper_name: str, indent: str = "        ") -> Optional[str]:
    sig = HELPER_SIGNATURES.get(helper_name)
    if sig is None:
        return None
    lines = [
        f'{indent}"""Return domain-specific metrics using standardized helper."""',
        f"{indent}{sig['import']}",
        f"{indent}return {helper_name}(",
    ]
    for param in sig["params"]:
        lines.append(f"{indent}    {param}={sig['defaults'][param]},")
    lines.append(f"{indent})")
    return "\n".join(lines)


def update_file(file_path: Path, helper_name: str, dry_run: bool = True) -> Tuple[bool, str]:
    try:
        content = _read_text(file_path)
    except OSError as exc:
        return False, f"Could not read file: {exc}"
    pattern = r"(def get_custom_metrics\s*\([^)]*\)[^:]*:)\s*\n((?:[ \t]+.*\n)*)"
    match = re.search(pattern, content)
    if not match:
        return False, "No get_custom_metrics method found"
    first_line = match.group(2).split("\n")[0] if match.group(2) else ""
    indent_match = re.match(r"^(\s+)", first_line)
    indent = indent_match.group(1) if indent_match else "        "
    new_impl = generate_helper_code(helper_name, indent)
    if new_impl is None:
        return False, f"Unknown helper: {helper_name}"
    new_method = f"{match.group(1)}\n{new_impl}\n"
    new_content = content[: match.start()] + new_method + content[match.end() :]
    if dry_run:
        return True, f"Would update with {helper_name}"
    file_path.write_text(new_content)
    return True, f"Updated with {helper_name}"


def iter_benchmark_files(root: Path, chapter: Optional[int] = None) -> Iterable[Path]:
    if chapter is not None:
        chapter_dir = root / f"ch{chapter:02d}"
        if chapter_dir.exists():
            yield from sorted(chapter_dir.glob("*.py"))
        return
    for chapter_dir in sorted(root.glob("ch*")):
        if chapter_dir.is_dir():
            yield from sorted(chapter_dir.glob("*.py"))
    labs_root = root / "labs"
    if labs_root.exists():
        for path in sorted(labs_root.rglob("*.py")):
            yield path


def scan_directory(root: Path, chapter: Optional[int] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for file_path in iter_benchmark_files(root, chapter=chapter):
        try:
            content = _read_text(file_path)
        except OSError:
            continue
        if not _is_benchmark_candidate(file_path, content):
            continue
        analysis = analyze_get_custom_metrics(file_path, root)
        ch = get_chapter_from_path(file_path)
        helper = CHAPTER_METRIC_HELPERS.get(ch)
        can_use_helper = (
            helper is not None
            and analysis.get("classification") in {"none", "empty", "phantom"}
            and not analysis.get("uses_helper")
            and not analysis.get("inherits_method")
            and not analysis.get("is_standalone")
            and not analysis.get("is_alias")
        )
        results.append(
            {
                "path": file_path,
                "chapter": ch,
                "analysis": analysis,
                "helper": helper,
                "can_use_helper": can_use_helper,
            }
        )
    return results


def audit_repo_custom_metrics(root: Optional[Path] = None, chapter: Optional[int] = None) -> List[Dict[str, Any]]:
    return scan_directory(root or get_repo_root(), chapter=chapter)


def validate_file_metrics(file_path: Path, root: Path) -> Dict[str, Any]:
    analysis = analyze_get_custom_metrics(file_path, root)
    classification = analysis.get("classification")
    valid = classification not in {"phantom", "error"}
    issues: List[str] = []
    if classification == "phantom":
        issues.extend(analysis.get("phantom_reasons", []))
    elif classification == "error":
        issues.append(analysis.get("error", "Unknown error"))
    return {
        "valid": valid,
        "issues": issues,
        "classification": classification,
        "helper_names": analysis.get("helper_names", []),
    }


def print_analysis(results: Sequence[Dict[str, Any]], verbose: bool = False, show_standalone: bool = False) -> None:
    counts: Dict[str, int] = {}
    for entry in results:
        classification = entry["analysis"]["classification"]
        counts[classification] = counts.get(classification, 0) + 1

    print("=" * 70)
    print("get_custom_metrics() Audit Summary")
    print("=" * 70)
    print(f"Total benchmark candidates scanned: {len(results)}")
    for key in [
        "helper-backed",
        "real",
        "inherited",
        "alias",
        "standalone",
        "none",
        "empty",
        "phantom",
        "missing",
        "error",
    ]:
        if counts.get(key):
            print(f"  {key:14s}: {counts[key]}")

    phantoms = [entry for entry in results if entry["analysis"]["classification"] == "phantom"]
    if phantoms:
        print()
        print("Phantom metrics:")
        print("-" * 70)
        for entry in phantoms[:40]:
            reasons = "; ".join(entry["analysis"]["phantom_reasons"][:3])
            print(f"  {entry['path']}: {reasons}")
        if len(phantoms) > 40:
            print(f"  ... and {len(phantoms) - 40} more")

    if show_standalone:
        standalone = [entry for entry in results if entry["analysis"]["classification"] == "standalone"]
        if standalone:
            print()
            print("Standalone scripts:")
            print("-" * 70)
            for entry in standalone:
                print(f"  {entry['path']}")

    if verbose:
        print()
        print("By chapter:")
        print("-" * 70)
        chapter_stats: Dict[Optional[int], Dict[str, int]] = {}
        for entry in results:
            chapter = entry["chapter"]
            stats = chapter_stats.setdefault(chapter, {})
            classification = entry["analysis"]["classification"]
            stats[classification] = stats.get(classification, 0) + 1
        for chapter in sorted(chapter_stats, key=lambda value: -1 if value is None else value):
            stats = chapter_stats[chapter]
            label = "labs" if chapter is None else f"ch{chapter:02d}"
            helper = CHAPTER_METRIC_HELPERS.get(chapter) if chapter is not None else None
            summary = ", ".join(f"{name}={count}" for name, count in sorted(stats.items()))
            print(f"  {label}: {summary} (default helper: {helper or 'custom'})")


def apply_updates(results: Sequence[Dict[str, Any]], dry_run: bool = True, specific_file: Optional[str] = None) -> int:
    updates = 0
    for entry in results:
        if not entry["can_use_helper"]:
            continue
        path = entry["path"]
        if specific_file and str(path) != specific_file and path.name != specific_file:
            continue
        helper = entry["helper"]
        if helper is None:
            continue
        success, message = update_file(path, helper, dry_run=dry_run)
        if success:
            updates += 1
            status = "[DRY-RUN]" if dry_run else "[UPDATED]"
            print(f"{status} {path}: {message}")
        else:
            print(f"[SKIPPED] {path}: {message}")
    return updates


def run_validation(results: Sequence[Dict[str, Any]]) -> int:
    invalid: List[Tuple[Path, Dict[str, Any]]] = []
    for entry in results:
        validation = validate_file_metrics(entry["path"], get_repo_root())
        if not validation["valid"]:
            invalid.append((entry["path"], validation))
    print("=" * 70)
    print("Metric Validation Report")
    print("=" * 70)
    print(f"Files validated: {len(results)}")
    print(f"Files with issues: {len(invalid)}")
    if invalid:
        print()
        for path, validation in invalid[:40]:
            issues = "; ".join(validation["issues"])
            print(f"  {path}: {issues}")
        if len(invalid) > 40:
            print(f"  ... and {len(invalid) - 40} more")
        return 1
    print("✅ All benchmark candidates passed validation.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and update get_custom_metrics() implementations")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--chapter", type=int, help="Focus on specific chapter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--apply", action="store_true", help="Apply updates (not dry-run)")
    parser.add_argument("--file", type=str, help="Apply to specific file only")
    parser.add_argument("--show-suggestion", type=str, help="Show suggestion for a file")
    parser.add_argument("--show-standalone", action="store_true", help="Show standalone scripts that need conversion")
    parser.add_argument("--validate", action="store_true", help="Validate metric quality in all files")
    args = parser.parse_args()

    root = get_repo_root()

    if args.show_suggestion:
        file_path = root / args.show_suggestion
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        analysis = analyze_get_custom_metrics(file_path, root)
        chapter = get_chapter_from_path(file_path)
        helper = CHAPTER_METRIC_HELPERS.get(chapter)
        print(f"File: {file_path}")
        print(f"Chapter: {chapter}")
        print(f"Classification: {analysis.get('classification')}")
        if analysis.get("phantom_reasons"):
            print("Phantom reasons:")
            for reason in analysis["phantom_reasons"]:
                print(f"  - {reason}")
        print()
        if helper is not None and analysis.get("classification") in {"none", "empty", "phantom"}:
            print(f"Suggested helper: {helper}")
            print()
            print("Generated implementation:")
            print("-" * 40)
            print("    def get_custom_metrics(self) -> Optional[dict]:")
            generated = generate_helper_code(helper, "        ")
            if generated is not None:
                print(generated)
        else:
            print("No automatic helper suggestion for this file.")
        return

    results = scan_directory(root, chapter=args.chapter)
    print_analysis(results, verbose=args.verbose, show_standalone=args.show_standalone)

    if args.apply or args.file:
        print()
        print("=" * 70)
        print("Applying Updates")
        print("=" * 70)
        dry_run = not args.apply
        count = apply_updates(results, dry_run=dry_run, specific_file=args.file)
        if dry_run:
            print()
            print(f"Would update {count} files. Use --apply to actually update.")
        else:
            print()
            print(f"Updated {count} files.")
        return

    if args.validate:
        print()
        sys.exit(run_validation(results))

    if not args.analyze:
        print()
        print("To validate metric quality:")
        print("  python core/scripts/update_custom_metrics.py --validate")


if __name__ == "__main__":
    main()
