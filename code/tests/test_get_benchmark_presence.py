"""Enforce that harness-discoverable Python benchmarks expose get_benchmark().

The benchmark harness imports each baseline_/optimized_ Python module and calls
`get_benchmark()` to construct a BaseBenchmark instance. This test ensures the
symbol is present (defined or re-exported) for every discoverable benchmark
module, matching the harness' non-recursive discovery behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]

from core.discovery import discover_all_chapters, should_ignore_benchmark_candidate  # noqa: E402


def _has_get_benchmark_symbol(path: Path) -> bool:
    """Return True if module defines or re-exports get_benchmark."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_benchmark":
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "get_benchmark" or alias.asname == "get_benchmark":
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "get_benchmark":
                    return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "get_benchmark":
                    return True
    return False


def _is_name_main_expr(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id == "__name__"
    if not isinstance(expr, ast.Call):
        return False
    if not isinstance(expr.func, ast.Attribute) or expr.func.attr != "get":
        return False
    if len(expr.args) != 1 or expr.keywords:
        return False
    if not isinstance(expr.args[0], ast.Constant) or expr.args[0].value != "__name__":
        return False
    globals_call = expr.func.value
    return (
        isinstance(globals_call, ast.Call)
        and isinstance(globals_call.func, ast.Name)
        and globals_call.func.id == "globals"
        and not globals_call.args
        and not globals_call.keywords
    )


def _is_main_guard_test(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not _is_name_main_expr(test.left):
        return False
    right = test.comparators[0]
    return (
        isinstance(right, ast.Constant)
        and right.value == "__main__"
        and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
    )


def _has_main_guard(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard_test(node.test):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(child, ast.If) and _is_main_guard_test(child.test)
            for child in ast.walk(node)
        ):
            helper_names.add(node.name)
        if (
            helper_names
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in helper_names
        ):
            return True
    return False


def _is_self_targeting_torchrun_wrapper(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return (
        "TorchrunLaunchSpec(" in text
        and "script_path=Path(__file__).resolve()" in text
        and "run_main_with_skip_status(main)" in text
    )


def test_all_discoverable_python_benchmarks_have_get_benchmark() -> None:
    missing: list[Path] = []
    for bench_dir in discover_all_chapters(REPO_ROOT):
        for path in list(bench_dir.glob("baseline_*.py")) + list(bench_dir.glob("optimized_*.py")):
            if should_ignore_benchmark_candidate(path):
                continue
            try:
                if not _has_get_benchmark_symbol(path):
                    missing.append(path)
            except SyntaxError:
                missing.append(path)

    if not missing:
        return

    lines = ["Missing get_benchmark() symbol in harness-discoverable benchmark modules:"]
    for path in sorted(missing):
        lines.append(f"  - {path.relative_to(REPO_ROOT)}")
    raise AssertionError("\n".join(lines))


def test_discoverable_python_benchmarks_do_not_define_main_guards() -> None:
    offenders: list[Path] = []
    for bench_dir in discover_all_chapters(REPO_ROOT):
        for path in list(bench_dir.glob("baseline_*.py")) + list(bench_dir.glob("optimized_*.py")):
            if should_ignore_benchmark_candidate(path):
                continue
            try:
                if (
                    _has_get_benchmark_symbol(path)
                    and _has_main_guard(path)
                    and not _is_self_targeting_torchrun_wrapper(path)
                ):
                    offenders.append(path)
            except SyntaxError:
                offenders.append(path)

    if not offenders:
        return

    lines = ["Discoverable benchmark modules must not define __main__ guards:"]
    for path in sorted(offenders):
        lines.append(f"  - {path.relative_to(REPO_ROOT)}")
    raise AssertionError("\n".join(lines))


def test_docs_do_not_suggest_direct_python_m_for_benchmark_modules() -> None:
    module_ref = re.compile(r"python -m ((?:ch\d{2}|labs(?:\.[A-Za-z0-9_]+)*)\.[A-Za-z0-9_\.]+)")
    paths = (
        [REPO_ROOT / "README.md"]
        + sorted(REPO_ROOT.glob("ch*/README.md"))
        + sorted(REPO_ROOT.glob("labs/**/README.md"))
        + [
            REPO_ROOT / "core" / "scripts" / "refresh_readmes.py",
            REPO_ROOT / "core" / "benchmark" / "examples.py",
        ]
    )

    offenders: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in module_ref.finditer(text):
            module_name = match.group(1)
            module_path = REPO_ROOT / Path(module_name.replace(".", "/")).with_suffix(".py")
            if not module_path.exists():
                continue
            if _has_get_benchmark_symbol(module_path):
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {module_name}")

    if not offenders:
        return

    lines = ["Docs/examples must route benchmark modules through compare.py or cli.aisp bench run:"]
    lines.extend(f"  - {entry}" for entry in offenders)
    raise AssertionError("\n".join(lines))
