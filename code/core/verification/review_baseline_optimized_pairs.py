#!/usr/bin/env python3
"""Review all baseline/optimized pairs for questionable practices.

Checks for:
- sleep() calls in benchmark code
- Artificial delays
- Mismatched workloads
- Unfair comparisons
- Missing synchronizations
- Other questionable practices
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

from core.discovery import discover_benchmark_pairs


FINDING_METADATA: Dict[str, Dict[str, str]] = {
    "error": {"issue_id": "AUDIT_FILE_READ_ERROR", "category": "tooling"},
    "sleep_call": {"issue_id": "BENCHMARK_SLEEP_CALL", "category": "timing"},
    "artificial_delay": {"issue_id": "BENCHMARK_ARTIFICIAL_DELAY", "category": "timing"},
    "suspicious_pattern": {"issue_id": "BENCHMARK_SUSPICIOUS_PATTERN", "category": "documentation"},
    "work_reduction": {"issue_id": "PAIR_WORKLOAD_MISMATCH", "category": "workload"},
    "sync_mismatch": {"issue_id": "PAIR_SYNC_MISMATCH", "category": "timing"},
    "seed_mismatch": {"issue_id": "PAIR_SEED_MISMATCH", "category": "environment"},
    "config_mismatch": {"issue_id": "PAIR_CONFIG_MISMATCH", "category": "config"},
    "precision_mismatch": {"issue_id": "PAIR_PRECISION_MISMATCH", "category": "workload"},
    "hot_path_extra_work": {"issue_id": "PAIR_HOT_PATH_EXTRA_WORK", "category": "hot_path"},
    "algorithmic_pair_mismatch": {"issue_id": "PAIR_ALGORITHMIC_MISMATCH", "category": "semantic"},
    "report_drift": {"issue_id": "PAIR_REVIEW_REPORT_DRIFT", "category": "documentation"},
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope_from_paths(baseline_path: Optional[Path], optimized_path: Optional[Path], file_label: str) -> str:
    for path in (baseline_path, optimized_path):
        if path is None:
            continue
        try:
            return path.parent.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
    if " vs " in file_label:
        lhs = file_label.split(" vs ", 1)[0]
        try:
            return Path(lhs).resolve().parent.relative_to(REPO_ROOT).as_posix()
        except Exception:
            return ""
    return ""


def _make_issue(
    *,
    file: str,
    issue_type: str,
    severity: str,
    message: str,
    baseline_path: Optional[Path] = None,
    optimized_path: Optional[Path] = None,
    evidence: Optional[Any] = None,
    status: str = "finding",
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = FINDING_METADATA.get(issue_type, {"issue_id": f"UNKNOWN_{issue_type.upper()}", "category": "unknown"})
    return {
        "file": file,
        "type": issue_type,
        "issue_id": metadata["issue_id"],
        "severity": severity,
        "category": metadata["category"],
        "scope": _scope_from_paths(baseline_path, optimized_path, file),
        "baseline_path": str(baseline_path) if baseline_path is not None else None,
        "optimized_path": str(optimized_path) if optimized_path is not None else None,
        "message": message,
        "evidence": evidence,
        "status": status,
        "skip_reason": skip_reason,
    }


@dataclass
class ReviewReport:
    timestamp: str
    chapters: List[str]
    total_pairs: int
    findings: List[Dict[str, Any]] = field(default_factory=list)

    def severity_counts(self) -> Dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in self.findings:
            severity = str(finding.get("severity", "low"))
            if severity not in counts:
                counts[severity] = 0
            counts[severity] += 1
        return counts

    def pair_statuses(self) -> Dict[str, str]:
        statuses = {}
        for finding in self.findings:
            baseline_path = finding.get("baseline_path")
            optimized_path = finding.get("optimized_path")
            if not baseline_path or not optimized_path:
                continue
            key = f"{_benchmark_scope_key(Path(baseline_path))}:{_benchmark_example_name(Path(optimized_path))}"
            statuses[key] = "FLAG"
        return statuses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "chapters": self.chapters,
            "summary": {
                "total_pairs": self.total_pairs,
                "total_findings": len(self.findings),
                "severity_counts": self.severity_counts(),
            },
            "pair_statuses": self.pair_statuses(),
            "findings": self.findings,
        }


def _get_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _get_call_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _get_self_method_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
        return node.func.attr
    return None


def _eval_numeric_expr(node: ast.AST, attr_values: Dict[str, float], local_values: Dict[str, float]) -> Optional[float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        return None
    if isinstance(node, ast.Name):
        return local_values.get(node.id)
    if isinstance(node, ast.Attribute):
        attr_name = _get_attr_name(node)
        if attr_name is not None:
            return attr_values.get(attr_name)
        return None
    if isinstance(node, ast.UnaryOp):
        operand = _eval_numeric_expr(node.operand, attr_values, local_values)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_expr(node.left, attr_values, local_values)
        right = _eval_numeric_expr(node.right, attr_values, local_values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right if right != 0 else None
        if isinstance(node.op, ast.FloorDiv):
            return float(int(left) // int(right)) if right != 0 else None
        if isinstance(node.op, ast.Mod):
            return left % right if right != 0 else None
        if isinstance(node.op, ast.Pow):
            return left ** right
        return None
    if isinstance(node, ast.Call):
        func_name = _get_call_attr_name(node.func)
        if func_name in {"float", "int"} and len(node.args) == 1:
            arg_val = _eval_numeric_expr(node.args[0], attr_values, local_values)
            if arg_val is None:
                return None
            return float(arg_val) if func_name == "float" else float(int(arg_val))
        if func_name in {"max", "min"} and node.args:
            evaluated_args = [
                _eval_numeric_expr(arg, attr_values, local_values)
                for arg in node.args
            ]
            if any(val is None for val in evaluated_args):
                return None
            return float(max(evaluated_args) if func_name == "max" else min(evaluated_args))
        return None
    if isinstance(node, ast.IfExp):
        # try both branches; only keep result if both resolve to same value
        true_val = _eval_numeric_expr(node.body, attr_values, local_values)
        false_val = _eval_numeric_expr(node.orelse, attr_values, local_values)
        if true_val is not None and false_val is not None and true_val == false_val:
            return true_val
        return None
    return None


def _maybe_extract_metadata_from_call(
    call: ast.Call,
    attr_values: Dict[str, float],
    local_values: Dict[str, float],
) -> Optional[Dict[str, float]]:
    if isinstance(call.func, ast.Name) and call.func.id == "WorkloadMetadata":
        keywords = call.keywords
    elif isinstance(call.func, ast.Attribute) and call.func.attr == "register_workload_metadata":
        keywords = call.keywords
    else:
        return None
    metadata: Dict[str, float] = {}
    for kw in keywords:
        if kw.arg is None:
            continue
        value = _eval_numeric_expr(kw.value, attr_values, local_values)
        if value is not None:
            metadata[kw.arg] = value
    return metadata or None


class _MethodAnalyzer:
    def __init__(self, attr_values: Dict[str, float]):
        self.attr_values = dict(attr_values)
        self.local_values: Dict[str, float] = {}
        self.metadata: List[Dict[str, float]] = []

    def process(self, func_def: ast.FunctionDef) -> None:
        self._process_block(func_def.body)

    def _process_block(self, statements: List[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                self._handle_assign(stmt)
            elif isinstance(stmt, ast.AnnAssign):
                self._handle_ann_assign(stmt)
            elif isinstance(stmt, ast.Expr):
                self._handle_expr(stmt.value)
            elif isinstance(stmt, ast.If):
                self._process_block(stmt.body)
                self._process_block(stmt.orelse)
            elif isinstance(stmt, ast.With):
                self._process_block(stmt.body)
            elif isinstance(stmt, ast.For):
                self._process_block(stmt.body)
                self._process_block(stmt.orelse)
            elif isinstance(stmt, ast.While):
                self._process_block(stmt.body)
                self._process_block(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                self._process_block(stmt.body)
                for handler in stmt.handlers:
                    self._process_block(handler.body)
                self._process_block(stmt.orelse)
                self._process_block(stmt.finalbody)

    def _handle_assign(self, node: ast.Assign) -> None:
        self._handle_expr(node.value)
        value = _eval_numeric_expr(node.value, self.attr_values, self.local_values)
        if value is None:
            return
        for target in node.targets:
            self._assign_target(target, value)

    def _handle_ann_assign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        self._handle_expr(node.value)
        value = _eval_numeric_expr(node.value, self.attr_values, self.local_values)
        if value is None:
            return
        self._assign_target(node.target, value)

    def _assign_target(self, target: ast.AST, value: float) -> None:
        if isinstance(target, ast.Attribute):
            attr_name = _get_attr_name(target)
            if attr_name is not None:
                self.attr_values[attr_name] = value
        elif isinstance(target, ast.Name):
            self.local_values[target.id] = value

    def _handle_expr(self, expr: ast.AST) -> None:
        if isinstance(expr, ast.Call):
            metadata = _maybe_extract_metadata_from_call(expr, self.attr_values, self.local_values)
            if metadata:
                self.metadata.append(metadata)


class _ClassAnalyzer:
    def __init__(self):
        self.attr_values: Dict[str, float] = {}
        self.metadata: List[Dict[str, float]] = []

    def process(self, node: ast.ClassDef) -> None:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in {"__init__", "setup"}:
                analyzer = _MethodAnalyzer(self.attr_values)
                analyzer.process(item)
                self.attr_values.update(analyzer.attr_values)
                self.metadata.extend(analyzer.metadata)


def extract_workload_metadata_from_source(content: str) -> Optional[Dict[str, float]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    extractor = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_analyzer = _ClassAnalyzer()
            class_analyzer.process(node)
            extractor.extend(class_analyzer.metadata)
    return extractor[0] if extractor else None


DECLARED_WORKLOAD_KEYS = (
    "requests_per_iteration",
    "tokens_per_iteration",
    "samples_per_iteration",
    "bytes_per_iteration",
    "custom_units_per_iteration",
    "batch_size",
    "N",
    "size",
    "hidden_size",
    "hidden_dim",
    "seq_len",
    "steps",
    "num_loops",
    "num_layers",
    "total_tokens",
    "inner_iterations",
)


def extract_declared_workload_signature_from_source(content: str) -> Optional[Dict[str, float]]:
    metadata = extract_workload_metadata_from_source(content)
    if metadata:
        return metadata

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_analyzer = _ClassAnalyzer()
        class_analyzer.process(node)
        declared = {
            key: float(class_analyzer.attr_values[key])
            for key in DECLARED_WORKLOAD_KEYS
            if key in class_analyzer.attr_values
        }
        if declared:
            return declared
    return None


class _TimedPathSyncAnalyzer(ast.NodeVisitor):
    def __init__(self, methods: Dict[str, ast.FunctionDef]):
        self.methods = methods
        self.visited_methods: Set[str] = set()
        self.sync_calls = 0

    def analyze(self, method_name: str) -> int:
        self._visit_method(method_name)
        return self.sync_calls

    def _visit_method(self, method_name: str) -> None:
        if method_name in self.visited_methods:
            return
        func_def = self.methods.get(method_name)
        if func_def is None:
            return
        self.visited_methods.add(method_name)
        for stmt in func_def.body:
            self.visit(stmt)

    def visit_Call(self, node: ast.Call) -> None:
        helper_name = _get_self_method_name(node)
        if helper_name is not None and helper_name in self.methods:
            self._visit_method(helper_name)

        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in {"synchronize", "wait"}:
                self.sync_calls += 1
        self.generic_visit(node)


def extract_timed_path_sync_count(content: str) -> int:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0

    total = 0
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            item.name: item
            for item in node.body
            if isinstance(item, ast.FunctionDef)
        }
        if "benchmark_fn" not in methods:
            continue
        analyzer = _TimedPathSyncAnalyzer(methods)
        total += analyzer.analyze("benchmark_fn")
    return total


def _merge_seed_info(*seed_infos: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[int, ...]]:
    manual: Set[int] = set()
    cuda: Set[int] = set()
    for seed_info in seed_infos:
        manual.update(seed_info.get("manual", ()))
        cuda.update(seed_info.get("cuda", ()))
    return {
        "manual": tuple(sorted(manual)),
        "cuda": tuple(sorted(cuda)),
    }


def _seed_info_from_content(content: str) -> Dict[str, Tuple[int, ...]]:
    manual = tuple(sorted({int(value) for value in re.findall(r"torch\.manual_seed\((\d+)\)", content)}))
    cuda = tuple(sorted({int(value) for value in re.findall(r"torch\.cuda\.manual_seed_all\((\d+)\)", content)}))
    return {"manual": manual, "cuda": cuda}


def _resolve_local_module_path(importing_file: Path, module: Optional[str], level: int) -> Optional[Path]:
    if level > 0:
        base_dir = importing_file.parent
        for _ in range(level - 1):
            base_dir = base_dir.parent
        search_roots = [base_dir]
    else:
        search_roots = [REPO_ROOT, importing_file.parent]

    rel_module = Path(*module.split(".")) if module else Path()
    for root in search_roots:
        candidates = []
        if rel_module.parts:
            candidates.append((root / rel_module).with_suffix(".py"))
            candidates.append(root / rel_module / "__init__.py")
        else:
            candidates.append(root / "__init__.py")
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    return None


def _extract_local_import_map(file_path: Path, tree: ast.AST) -> Dict[str, Path]:
    import_map: Dict[str, Path] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_path = _resolve_local_module_path(file_path, node.module, node.level)
        if module_path is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            import_map[alias.asname or alias.name] = module_path
    return import_map


def extract_seed_info(
    file_path: Path,
    content: str,
    *,
    _visited: Optional[Set[Path]] = None,
) -> Dict[str, Tuple[int, ...]]:
    visited = _visited or set()
    resolved_path = file_path.resolve()
    if resolved_path in visited:
        return {"manual": (), "cuda": ()}
    visited.add(resolved_path)

    seed_info = _seed_info_from_content(content)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return seed_info

    import_map = _extract_local_import_map(file_path, tree)
    inherited_infos: List[Dict[str, Tuple[int, ...]]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name: Optional[str] = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if not base_name:
                continue
            imported_path = import_map.get(base_name)
            if imported_path is None:
                continue
            try:
                imported_content = imported_path.read_text(encoding="utf-8")
            except OSError:
                continue
            inherited_infos.append(
                extract_seed_info(imported_path, imported_content, _visited=visited)
            )

    return _merge_seed_info(seed_info, *inherited_infos)


def _eval_literal_expr(node: ast.AST) -> Optional[object]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp):
        operand = _eval_literal_expr(node.operand)
        if not isinstance(operand, (int, float)):
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
    return None


def extract_benchmark_config_from_source(content: str) -> Optional[Dict[str, object]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "get_config":
                continue
            for child in ast.walk(item):
                if not isinstance(child, ast.Call):
                    continue
                if not isinstance(child.func, ast.Name) or child.func.id != "BenchmarkConfig":
                    continue
                config: Dict[str, object] = {}
                for kw in child.keywords:
                    if kw.arg is None:
                        continue
                    value = _eval_literal_expr(kw.value)
                    if value is not None:
                        config[kw.arg] = value
                return config or None
    return None


_DTYPE_PATTERN = re.compile(r"torch\.(float16|float32|bfloat16|int8)")


def extract_primary_dtype(content: str) -> Optional[str]:
    preamble = content.split("def capture_verification_payload", 1)[0]
    matches = _DTYPE_PATTERN.findall(preamble)
    if not matches:
        return None
    return Counter(matches).most_common(1)[0][0]


class _TimedPathFeatureAnalyzer(ast.NodeVisitor):
    def __init__(self, methods: Dict[str, ast.FunctionDef]):
        self.methods = methods
        self.visited_methods: Set[str] = set()
        self.clone_calls = 0
        self.dtype_casts = 0
        self.model_calls = 0

    def analyze(self, method_name: str) -> Dict[str, int]:
        self._visit_method(method_name)
        return {
            "clone_calls": self.clone_calls,
            "dtype_casts": self.dtype_casts,
            "model_calls": self.model_calls,
        }

    def _visit_method(self, method_name: str) -> None:
        if method_name in self.visited_methods:
            return
        func_def = self.methods.get(method_name)
        if func_def is None:
            return
        self.visited_methods.add(method_name)
        for stmt in func_def.body:
            self.visit(stmt)

    def visit_Call(self, node: ast.Call) -> None:
        helper_name = _get_self_method_name(node)
        if helper_name is not None and helper_name in self.methods:
            self._visit_method(helper_name)

        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "clone":
                self.clone_calls += 1
            if node.func.attr == "to" and any(kw.arg == "dtype" for kw in node.keywords):
                self.dtype_casts += 1
            if node.func.attr in {"float", "half", "bfloat16"}:
                self.dtype_casts += 1
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.func.attr in {"model", "compiled_model"}:
                self.model_calls += 1
        self.generic_visit(node)


def extract_hot_path_features(content: str) -> Dict[str, int]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"clone_calls": 0, "dtype_casts": 0, "model_calls": 0}

    total = {"clone_calls": 0, "dtype_casts": 0, "model_calls": 0}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
        if "benchmark_fn" not in methods:
            continue
        analyzer = _TimedPathFeatureAnalyzer(methods)
        features = analyzer.analyze("benchmark_fn")
        for key, value in features.items():
            total[key] += value
    return total


def extract_algorithmic_markers(content: str) -> Dict[str, object]:
    benchmark_prefix = content.split("def capture_verification_payload", 1)[0]
    route_modes = sorted(set(match for match in re.findall(r'route_mode\s*=\s*"([^"]+)"', content) if match))
    return {
        "route_modes": route_modes,
        "blockwise_decode": bool(re.search(r"range\s*\(\s*0\s*,\s*seq_len\s*,", benchmark_prefix)),
    }


def _uses_cuda_graphs(content: str) -> bool:
    return any(
        marker in content
        for marker in ("torch.cuda.CUDAGraph", ".graph.replay()", "torch.cuda.graph(")
    )


def _calls_super_setup(content: str) -> bool:
    return "super().setup()" in content


def _is_intentional_precision_target(baseline_path: Path, opt_path: Path) -> bool:
    keywords = ("precisionfp", "quantization", "fp8", "fp4", "mixed")
    names = (baseline_path.stem, opt_path.stem)
    return any(any(keyword in name for keyword in keywords) for name in names)


def _ignores_precision_flags_in_signature(content: str) -> bool:
    return bool(
        re.search(
            r"signature_equivalence_ignore_fields\s*=\s*.*precision_flags",
            content,
            re.DOTALL,
        )
    )


def _benchmark_example_name(path: Path) -> str:
    stem = path.stem
    for prefix in ("baseline_", "optimized_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _benchmark_scope_key(path: Path) -> str:
    try:
        return path.parent.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return ""


def _is_informational_benchmark(path: Path) -> bool:
    try:
        from core.harness.run_benchmarks import INFORMATIONAL_BENCHMARKS
    except Exception:
        return False
    scope = _benchmark_scope_key(path)
    example_name = _benchmark_example_name(path)
    scope_candidates = [scope]
    if scope:
        scope_candidates.append(Path(scope).name)
    return any(example_name in INFORMATIONAL_BENCHMARKS.get(candidate, set()) for candidate in scope_candidates)


def dedupe_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for issue in issues:
        key = (issue["type"], issue["file"], issue["message"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


class CodeReviewer:
    """Review code for questionable practices."""
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
    
    def check_file(self, file_path: Path, pair_type: str) -> List[Dict[str, Any]]:
        """Check a single file for issues."""
        file_issues = []
        
        try:
            content = file_path.read_text()
        except Exception as e:
            file_issues.append(
                _make_issue(
                    file=str(file_path),
                    issue_type="error",
                    severity="high",
                    message=f"Could not read file: {e}",
                    baseline_path=file_path if pair_type == "baseline" else None,
                    optimized_path=file_path if pair_type == "optimized" else None,
                )
            )
            return file_issues
        
        # Check for sleep calls
        if re.search(r'\btime\.sleep\s*\(', content):
            # Check if it's in benchmark_fn or setup/teardown
            if self._is_in_benchmark_code(content, 'time.sleep'):
                file_issues.append(
                    _make_issue(
                        file=str(file_path),
                        issue_type="sleep_call",
                        severity="critical",
                        message="time.sleep() found in benchmark code - this artificially inflates timing",
                        baseline_path=file_path if pair_type == "baseline" else None,
                        optimized_path=file_path if pair_type == "optimized" else None,
                    )
                )
        
        # Check for artificial delays
        delay_patterns = [
            (r'sleep\s*\(\s*[\d.]+\s*\)', "sleep() call"),
            (r'time\.sleep\s*\(', "time.sleep() call"),
            (r'asyncio\.sleep\s*\(', "asyncio.sleep() call"),
        ]
        for pattern, desc in delay_patterns:
            if re.search(pattern, content):
                if self._is_in_benchmark_code(content, pattern):
                    file_issues.append(
                        _make_issue(
                            file=str(file_path),
                            issue_type="artificial_delay",
                            severity="critical",
                            message=f"{desc} found in benchmark code",
                            baseline_path=file_path if pair_type == "baseline" else None,
                            optimized_path=file_path if pair_type == "optimized" else None,
                        )
                    )
        
        # Check for missing synchronizations in optimized but present in baseline
        # This is harder - we'll do pair-wise comparison
        
        # Only flag explicit deception markers in source comments, not docstrings or CLI help.
        suspicious_comment_pattern = re.compile(
            r'#.*\b(?:fake benchmark|dummy benchmark|mock benchmark|benchmark is fake|benchmark is dummy|benchmark is mock)\b',
            re.IGNORECASE,
        )
        for line in content.splitlines():
            if suspicious_comment_pattern.search(line):
                file_issues.append(
                    _make_issue(
                        file=str(file_path),
                        issue_type="suspicious_pattern",
                        severity="medium",
                        message="Suspicious comment suggesting benchmark deception",
                        baseline_path=file_path if pair_type == "baseline" else None,
                        optimized_path=file_path if pair_type == "optimized" else None,
                    )
                )
                break
        
        return file_issues
    
    def _is_in_benchmark_code(self, content: str, pattern: str) -> bool:
        """Check if pattern appears in benchmark code (setup/benchmark_fn/teardown)."""
        # Simple heuristic: check if it's in a method that's likely benchmark code
        # Look for def setup, def benchmark_fn, def teardown
        lines = content.split('\n')
        in_benchmark_method = False
        method_depth = 0
        
        for i, line in enumerate(lines):
            # Track method definitions
            if re.match(r'\s*def\s+(setup|benchmark_fn|teardown)', line):
                in_benchmark_method = True
                method_depth = len(line) - len(line.lstrip())
            elif in_benchmark_method:
                # Check if we've left the method (dedented)
                current_depth = len(line) - len(line.lstrip()) if line.strip() else method_depth
                if current_depth <= method_depth and line.strip() and not line.strip().startswith('#'):
                    in_benchmark_method = False
                
                # Check if pattern is in this method
                if re.search(pattern, line):
                    return True
        
        return False
    
    def compare_pair(self, baseline_path: Path, optimized_paths: List[Path]) -> List[Dict[str, Any]]:
        """Compare a baseline/optimized pair for fairness."""
        pair_issues = []
        
        try:
            baseline_content = baseline_path.read_text()
        except Exception as e:
            pair_issues.append(
                _make_issue(
                    file=f"{baseline_path} (pair)",
                    issue_type="error",
                    severity="high",
                    message=f"Could not read baseline: {e}",
                    baseline_path=baseline_path,
                )
            )
            return pair_issues
        
        for opt_path in optimized_paths:
            try:
                opt_content = opt_path.read_text()
            except Exception as e:
                pair_issues.append(
                    _make_issue(
                        file=f"{opt_path} (pair)",
                        issue_type="error",
                        severity="high",
                        message=f"Could not read optimized: {e}",
                        optimized_path=opt_path,
                    )
                )
                continue

            if _is_informational_benchmark(opt_path):
                continue
            
            # Check for declared workload mismatches only when we can prove them.
            baseline_workload = self._extract_workload_info(baseline_content)
            opt_workload = self._extract_workload_info(opt_content)

            if baseline_workload and opt_workload and not self._workloads_similar(baseline_workload, opt_workload):
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="work_reduction",
                        severity="medium",
                        message=(
                            "Declared workload differs between baseline and optimized: "
                            f"baseline={baseline_workload}, optimized={opt_workload}"
                        ),
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"baseline_workload": baseline_workload, "optimized_workload": opt_workload},
                    )
                )
            
            # Check for synchronization mismatches
            baseline_syncs = self._count_synchronizations(baseline_content)
            opt_syncs = self._count_synchronizations(opt_content)
            
            # Optimized should have same or fewer syncs (not more, which would slow it down unfairly)
            if opt_syncs > baseline_syncs * 1.5:  # Allow some variance
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="sync_mismatch",
                        severity="low",
                        message=f"Optimized has more synchronizations ({opt_syncs} vs {baseline_syncs}) - may be unfair",
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"baseline_syncs": baseline_syncs, "optimized_syncs": opt_syncs},
                    )
                )

            baseline_seed = extract_seed_info(baseline_path, baseline_content)
            opt_seed = extract_seed_info(opt_path, opt_content)
            if baseline_seed != opt_seed and not (_calls_super_setup(baseline_content) or _calls_super_setup(opt_content)):
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="seed_mismatch",
                        severity="high",
                        message=f"Seed setup differs: baseline={baseline_seed}, optimized={opt_seed}",
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"baseline_seed": baseline_seed, "optimized_seed": opt_seed},
                    )
                )

            baseline_config = extract_benchmark_config_from_source(baseline_content) or {}
            opt_config = extract_benchmark_config_from_source(opt_content) or {}
            precision_target = (
                _is_intentional_precision_target(baseline_path, opt_path)
                or _ignores_precision_flags_in_signature(baseline_content)
                or _ignores_precision_flags_in_signature(opt_content)
            )
            config_keys = {
                "iterations",
                "warmup",
                "adaptive_iterations",
                "enable_profiling",
                "enable_ncu",
                "enable_nsys",
            }
            config_diffs = {
                key: (baseline_config.get(key), opt_config.get(key))
                for key in config_keys
                if baseline_config.get(key) != opt_config.get(key)
                and key in baseline_config
                and key in opt_config
            }
            if config_diffs:
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="config_mismatch",
                        severity="high",
                        message=f"BenchmarkConfig differs: {config_diffs}",
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"config_diffs": config_diffs},
                    )
                )

            baseline_dtype = extract_primary_dtype(baseline_content)
            opt_dtype = extract_primary_dtype(opt_content)
            if (
                baseline_dtype
                and opt_dtype
                and baseline_dtype != opt_dtype
                and not precision_target
                and not (_calls_super_setup(baseline_content) or _calls_super_setup(opt_content))
            ):
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="precision_mismatch",
                        severity="high",
                        message=f"Primary compute dtype differs: baseline=torch.{baseline_dtype}, optimized=torch.{opt_dtype}",
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"baseline_dtype": baseline_dtype, "optimized_dtype": opt_dtype},
                    )
                )

            baseline_hot = extract_hot_path_features(baseline_content)
            opt_hot = extract_hot_path_features(opt_content)
            hot_diffs = {
                key: (baseline_hot[key], opt_hot[key])
                for key in baseline_hot
                if opt_hot[key] > baseline_hot[key]
            }
            if precision_target:
                hot_diffs = {
                    key: value
                    for key, value in hot_diffs.items()
                    if key != "dtype_casts"
                }
            if hot_diffs and not (_uses_cuda_graphs(baseline_content) or _uses_cuda_graphs(opt_content)):
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="hot_path_extra_work",
                        severity="medium",
                        message=f"Hot-path operations differ: {hot_diffs}",
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"hot_path_diffs": hot_diffs},
                    )
                )

            baseline_algo = extract_algorithmic_markers(baseline_content)
            opt_algo = extract_algorithmic_markers(opt_content)
            algo_diffs = {
                key: (baseline_algo[key], opt_algo[key])
                for key in baseline_algo
                if baseline_algo[key] != opt_algo[key]
            }
            route_mode_target = "routing" in baseline_path.stem or "routing" in opt_path.stem
            if route_mode_target and set(algo_diffs) == {"route_modes"}:
                algo_diffs = {}
            if algo_diffs:
                pair_issues.append(
                    _make_issue(
                        file=f"{baseline_path} vs {opt_path}",
                        issue_type="algorithmic_pair_mismatch",
                        severity="high",
                        message=f"Pair semantics differ beyond execution strategy: {algo_diffs}",
                        baseline_path=baseline_path,
                        optimized_path=opt_path,
                        evidence={"algorithmic_diffs": algo_diffs},
                    )
                )
        
        return pair_issues
    
    def _extract_workload_info(self, content: str) -> Optional[Dict[str, float]]:
        """Extract workload information using metadata when available."""
        metadata = extract_declared_workload_signature_from_source(content)
        if metadata:
            return metadata
        return self._extract_workload_info_from_regex(content)

    def _extract_workload_info_from_regex(self, content: str) -> Optional[Dict[str, float]]:
        workload = {}
        patterns = [
            (r'self\.N\s*=\s*(\d+)', 'N'),
            (r'self\.batch_size\s*=\s*(\d+)', 'batch_size'),
            (r'self\.size\s*=\s*(\d+)', 'size'),
            (r'self\.hidden_size\s*=\s*(\d+)', 'hidden_size'),
            (r'self\.hidden_dim\s*=\s*(\d+)', 'hidden_dim'),
            (r'self\.seq_len\s*=\s*(\d+)', 'seq_len'),
            (r'self\.steps\s*=\s*(\d+)', 'steps'),
            (r'self\.num_loops\s*=\s*(\d+)', 'num_loops'),
            (r'self\.num_layers\s*=\s*(\d+)', 'num_layers'),
            (r'self\.total_tokens\s*=\s*(\d+)', 'total_tokens'),
            (r'self\.inner_iterations\s*=\s*(\d+)', 'inner_iterations'),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, content)
            if match:
                workload[key] = float(match.group(1))
        return workload if workload else None
    
    def _workloads_similar(self, w1: Dict[str, float], w2: Dict[str, float]) -> bool:
        """Check if workloads are similar (within 10%) using overlapping keys only."""
        comparable_keys = {
            "requests_per_iteration",
            "tokens_per_iteration",
            "samples_per_iteration",
            "bytes_per_iteration",
            "custom_units_per_iteration",
            "batch_size",
            "N",
            "size",
            "hidden_size",
            "hidden_dim",
            "seq_len",
            "steps",
            "num_loops",
            "num_layers",
            "total_tokens",
            "inner_iterations",
        }
        overlap = [k for k in comparable_keys if k in w1 and k in w2]
        if not overlap:
            return True
        for key in overlap:
            base = w1[key]
            opt = w2[key]
            denom = max(abs(base), abs(opt), 1e-9)
            if denom == 0:
                continue
            diff = abs(base - opt) / denom
            if diff > 0.1:
                return False
        return True
    
    def _count_synchronizations(self, content: str) -> int:
        """Count synchronization calls reachable from benchmark_fn()."""
        return extract_timed_path_sync_count(content)


def _discover_pairs_for_review(chapters: Optional[List[str]]) -> List[Tuple[Path, List[Path], str]]:
    requested_chapters = chapters or ["all"]
    all_pairs: List[Tuple[Path, List[Path], str]] = []
    seen: Set[Tuple[Path, Tuple[Path, ...], str]] = set()

    for chapter in requested_chapters:
        for baseline_path, optimized_paths, example_name in discover_benchmark_pairs(REPO_ROOT, chapter):
            key = (
                baseline_path.resolve(),
                tuple(path.resolve() for path in optimized_paths),
                example_name,
            )
            if key in seen:
                continue
            seen.add(key)
            all_pairs.append((baseline_path, optimized_paths, example_name))

    return all_pairs


def run_review(chapters: Optional[List[str]] = None) -> ReviewReport:
    requested_chapters = chapters or ["all"]
    pairs = _discover_pairs_for_review(chapters)
    reviewer = CodeReviewer()
    all_issues: List[Dict[str, Any]] = []

    for baseline_path, optimized_paths, _example_name in pairs:
        all_issues.extend(reviewer.check_file(baseline_path, "baseline"))
        for opt_path in optimized_paths:
            all_issues.extend(reviewer.check_file(opt_path, "optimized"))
        all_issues.extend(reviewer.compare_pair(baseline_path, optimized_paths))

    report = ReviewReport(
        timestamp=_utc_now_iso(),
        chapters=requested_chapters,
        total_pairs=len(pairs),
        findings=dedupe_issues(all_issues),
    )
    return report


def _render_review_markdown(report: ReviewReport) -> str:
    counts = report.severity_counts()
    lines = [
        "# Benchmark Pair Advisory Audit",
        "",
        f"- Timestamp: `{report.timestamp}`",
        f"- Scope: `{', '.join(report.chapters)}`",
        f"- Pairs reviewed: `{report.total_pairs}`",
        f"- Findings: `{len(report.findings)}`",
        "",
        "## Severity Summary",
        "",
        "| Severity | Count |",
        "|---|---:|",
        f"| critical | {counts.get('critical', 0)} |",
        f"| high | {counts.get('high', 0)} |",
        f"| medium | {counts.get('medium', 0)} |",
        f"| low | {counts.get('low', 0)} |",
        "",
    ]
    if not report.findings:
        lines.extend(
            [
                "## Findings",
                "",
                "No findings.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Findings",
            "",
            "| Issue ID | Severity | Scope | Type | File | Message |",
            "|---|---|---|---|---|---|",
        ]
    )
    for finding in report.findings:
        lines.append(
            "| {issue_id} | {severity} | {scope} | {type} | `{file}` | {message} |".format(
                issue_id=finding["issue_id"],
                severity=finding["severity"],
                scope=finding.get("scope") or "-",
                type=finding["type"],
                file=finding["file"],
                message=str(finding["message"]).replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_review_outputs(report: ReviewReport, output_dir: Path, *, write_json: bool, write_markdown: bool) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    if write_json:
        json_path = output_dir / "review_findings.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        paths["json"] = str(json_path)
    if write_markdown:
        markdown_path = output_dir / "review_findings.md"
        markdown_path.write_text(_render_review_markdown(report), encoding="utf-8")
        paths["markdown"] = str(markdown_path)
    return paths


def print_review_summary(report: ReviewReport) -> None:
    counts = report.severity_counts()
    print("Discovering benchmark pairs...")
    print(f"Found {report.total_pairs} pairs to review")
    print()
    print("=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)
    print(f"Scope:            {', '.join(report.chapters)}")
    print(f"Pairs reviewed:   {report.total_pairs}")
    print(f"Findings:         {len(report.findings)}")
    print(f"Critical issues:  {counts.get('critical', 0)}")
    print(f"High issues:      {counts.get('high', 0)}")
    print(f"Medium issues:    {counts.get('medium', 0)}")
    print(f"Low issues:       {counts.get('low', 0)}")
    if not report.findings:
        print("\n✓ No issues found!")
        return
    for severity in ("critical", "high", "medium", "low"):
        matches = [item for item in report.findings if item["severity"] == severity]
        if not matches:
            continue
        print(f"\n{severity.title()} findings: {len(matches)}")
        for finding in matches[:20]:
            print(f"  [{finding['type']}] {finding['file']}: {finding['message']}")
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more")


def main(argv: Optional[List[str]] = None) -> int:
    """Main review function."""
    parser = argparse.ArgumentParser(
        description="Review baseline/optimized benchmark pairs for fairness and questionable practices.",
    )
    parser.add_argument(
        "--chapter",
        action="append",
        dest="chapters",
        help="Limit the review to a chapter or lab path (repeatable). Example: --chapter ch12 --chapter labs/fullstack_cluster",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write structured JSON findings to --output-dir/review_findings.json",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Write markdown findings to --output-dir/review_findings.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for structured output artifacts",
    )
    args = parser.parse_args(argv)
    if (args.json or args.markdown) and args.output_dir is None:
        parser.error("--output-dir is required when --json or --markdown is set")

    report = run_review(args.chapters)
    if report.total_pairs == 0:
        print("No benchmark pairs found for the requested scope.")
    print_review_summary(report)

    if args.output_dir:
        written = write_review_outputs(
            report,
            args.output_dir,
            write_json=args.json,
            write_markdown=args.markdown,
        )
        for label, path in written.items():
            print(f"{label.title()} report written to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
