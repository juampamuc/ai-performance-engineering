"""Audit baseline/optimized benchmark alignment.

This script statically inspects each baseline/optimized pair discovered via
``core.discovery`` and emits a quick report showing whether the pair
shares the same optimizer, NVTX label, and key workload knobs (batch/seq/hidden).

It is intentionally lightweight (no benchmark execution) so we can run it on
any workstation before firing off the full benchmark suite.
"""

from __future__ import annotations

import ast
import json
import re
import importlib
import inspect
from dataclasses import dataclass, asdict
import argparse
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

repo_root = Path(__file__).resolve().parents[2]

from core.discovery import discover_all_chapters, discover_benchmarks
from core.profiling.nvtx_helper import canonicalize_nvtx_name

ATTR_ALIASES = {
    "batch_size": ("batch_size", "batch"),
    "seq_len": ("seq_len", "seq_length", "sequence_length", "max_seq_len"),
    "hidden_dim": ("hidden_dim", "hidden_size", "embed_dim", "d_model", "hidden", "hidden_dim_val", "head_dim"),
}
ATTR_PATTERNS = {
    key: [re.compile(rf"self\.{alias}\s*=\s*([^\n#]+)") for alias in aliases]
    for key, aliases in ATTR_ALIASES.items()
}

OPTIMIZER_PATTERN = re.compile(r"torch\.optim\.(?!Optimizer\b)([A-Za-z0-9_]+)\s*\(")
NVTX_PATTERN = re.compile(r"nvtx_range\(\s*['\"]([^'\"]+)['\"]")


@dataclass
class BenchmarkMetadata:
    path: str
    optimizer: str | None
    nvtx: List[str]
    batch_size: str | None
    seq_len: str | None
    hidden_dim: str | None


def extract_metadata(path: Path) -> BenchmarkMetadata:
    path = path.resolve()
    benchmark = load_benchmark_instance(path)

    optimizer = extract_optimizer(benchmark, path)
    nvtx = extract_nvtx_labels(benchmark, path)
    attrs = extract_attrs(benchmark, path)

    return BenchmarkMetadata(
        path=str(path.relative_to(repo_root)),
        optimizer=optimizer,
        nvtx=nvtx,
        batch_size=attrs["batch_size"],
        seq_len=attrs["seq_len"],
        hidden_dim=attrs["hidden_dim"],
    )


def load_benchmark_instance(path: Path) -> object | None:
    path = path.resolve()
    module_name = ".".join(path.relative_to(repo_root).with_suffix("").parts)
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    factory = getattr(module, "get_benchmark", None)
    if not callable(factory):
        return None
    try:
        return factory()
    except Exception:
        return None


def extract_attrs(benchmark: object | None, path: Path) -> Dict[str, str | None]:
    attrs: Dict[str, str | None] = {key: None for key in ATTR_PATTERNS}
    if benchmark is not None:
        for key, aliases in ATTR_ALIASES.items():
            for alias in aliases:
                if not hasattr(benchmark, alias):
                    continue
                value = normalize_scalar(getattr(benchmark, alias))
                if value is not None:
                    attrs[key] = value
                    break
    for key, patterns in ATTR_PATTERNS.items():
        if attrs[key] is not None:
            continue
        source_text = extract_class_source_text(benchmark, path)
        if source_text is None:
            source_text = path.read_text()
        for pattern in patterns:
            matches = pattern.findall(source_text)
            if matches:
                attrs[key] = matches[-1].strip()
                break
    return attrs


def extract_nvtx_labels(benchmark: object | None, path: Path) -> List[str]:
    search_regions: List[str] = []
    benchmark_fn_source = extract_benchmark_fn_source(benchmark)
    if benchmark_fn_source:
        search_regions.append(benchmark_fn_source)
    class_source = extract_class_source_text(benchmark, path)
    if class_source:
        search_regions.append(class_source)
    search_regions.append(path.read_text())

    labels: List[str] = []
    for region in search_regions:
        raw_nvtx = NVTX_PATTERN.findall(region)
        if raw_nvtx:
            labels = [canonicalize_nvtx_name(label) for label in raw_nvtx]
            break
    return labels


def extract_optimizer(benchmark: object | None, path: Path) -> str | None:
    if benchmark is not None:
        for cls in iter_repo_classes(type(benchmark)):
            optimizer = extract_optimizer_from_class(cls)
            if optimizer:
                return optimizer

    class_source = extract_class_source_text(benchmark, path)
    if class_source:
        match = OPTIMIZER_PATTERN.search(class_source)
        if match:
            return match.group(1)

    text = path.read_text()
    match = OPTIMIZER_PATTERN.search(text)
    return match.group(1) if match else None


def extract_benchmark_fn_source(benchmark: object | None) -> str | None:
    if benchmark is None:
        return None
    try:
        return inspect.getsource(type(benchmark).benchmark_fn)
    except Exception:
        return None


def extract_class_source_text(benchmark: object | None, path: Path) -> str | None:
    if benchmark is not None:
        for cls in iter_repo_classes(type(benchmark)):
            try:
                return inspect.getsource(cls)
            except Exception:
                continue
    return path.read_text()


def iter_repo_classes(benchmark_cls: type[Any]) -> Iterable[type[Any]]:
    for cls in benchmark_cls.__mro__:
        if cls is object:
            continue
        try:
            source_path = inspect.getsourcefile(cls)
        except Exception:
            continue
        if not source_path:
            continue
        source_file = Path(source_path).resolve()
        if repo_root in source_file.parents or source_file == repo_root:
            yield cls


def extract_optimizer_from_class(benchmark_cls: type[Any]) -> str | None:
    source_path = inspect.getsourcefile(benchmark_cls)
    if not source_path:
        return None
    module_path = Path(source_path).resolve()
    try:
        module_ast = parse_python_ast(module_path)
    except Exception:
        return None
    class_node = find_class_node(module_ast, benchmark_cls.__name__)
    if class_node is None:
        return None

    optimizer_aliases, module_aliases = collect_optimizer_imports(module_ast)
    visitor = OptimizerCallVisitor(optimizer_aliases, module_aliases)
    visitor.visit(class_node)
    return visitor.optimizers[0] if visitor.optimizers else None


@lru_cache(maxsize=None)
def parse_python_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def find_class_node(module_ast: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def collect_optimizer_imports(module_ast: ast.Module) -> tuple[Dict[str, str], Set[str]]:
    optimizer_aliases: Dict[str, str] = {}
    module_aliases: Set[str] = set()
    for node in module_ast.body:
        if isinstance(node, ast.ImportFrom) and node.module == "torch.optim":
            for alias in node.names:
                if alias.name == "Optimizer":
                    continue
                optimizer_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.optim":
                    module_aliases.add(alias.asname or "torch.optim")
    return optimizer_aliases, module_aliases


class OptimizerCallVisitor(ast.NodeVisitor):
    def __init__(self, optimizer_aliases: Dict[str, str], module_aliases: Set[str]) -> None:
        self.optimizer_aliases = optimizer_aliases
        self.module_aliases = module_aliases
        self.optimizers: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = resolve_call_name(node.func)
        optimizer = self.resolve_optimizer(name)
        if optimizer and optimizer not in self.optimizers:
            self.optimizers.append(optimizer)
        self.generic_visit(node)

    def resolve_optimizer(self, call_name: str | None) -> str | None:
        if not call_name:
            return None
        if call_name in self.optimizer_aliases:
            return self.optimizer_aliases[call_name]
        for module_alias in self.module_aliases:
            prefix = f"{module_alias}."
            if call_name.startswith(prefix):
                optimizer = call_name.split(".")[-1]
                return None if optimizer == "Optimizer" else optimizer
        if call_name.startswith("torch.optim."):
            optimizer = call_name.split(".")[-1]
            return None if optimizer == "Optimizer" else optimizer
        return None


def resolve_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = resolve_call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def normalize_scalar(value: object) -> str | None:
    if value is None or callable(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return str(value)
    return None


def compare_pair(baseline: BenchmarkMetadata, optimized: List[BenchmarkMetadata]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "baseline": asdict(baseline),
        "optimized": [asdict(meta) for meta in optimized],
        "mismatches": [],
    }

    def check(field: str, pretty_name: str) -> None:
        base_value = getattr(baseline, field)
        for meta in optimized:
            opt_value = getattr(meta, field)
            if not values_are_comparable(field, base_value, opt_value):
                continue
            if base_value != opt_value:
                result["mismatches"].append(
                    f"{pretty_name} mismatch: baseline={base_value} vs {meta.path}={opt_value}"
                )

    check("optimizer", "optimizer")
    check("batch_size", "batch_size")
    check("seq_len", "seq_len")
    check("hidden_dim", "hidden_dim")

    baseline_nvtx = baseline.nvtx[0] if baseline.nvtx else None
    for meta in optimized:
        opt_nvtx = meta.nvtx[0] if meta.nvtx else None
        if baseline_nvtx is None and opt_nvtx is None:
            continue
        if baseline_nvtx is None or opt_nvtx is None:
            result["mismatches"].append(
                f"NVTX label missing: baseline={baseline_nvtx} vs {meta.path}={opt_nvtx}"
            )

    result["severity"] = severity_label(len(result["mismatches"]))
    return result


def severity_label(mismatch_count: int) -> str:
    if mismatch_count == 0:
        return "clean"
    if mismatch_count == 1:
        return "low"
    if mismatch_count <= 3:
        return "medium"
    return "high"


def values_are_comparable(field: str, baseline_value: str | None, optimized_value: str | None) -> bool:
    if baseline_value is None or optimized_value is None:
        return False
    if field == "optimizer":
        return True
    return is_numeric_literal(baseline_value) and is_numeric_literal(optimized_value)


def is_numeric_literal(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def write_csv(report: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "chapter",
        "example",
        "baseline_path",
        "optimized_paths",
        "severity",
        "mismatch_count",
        "mismatches",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report:
            writer.writerow(
                {
                    "chapter": entry["chapter"],
                    "example": entry["example"],
                    "baseline_path": entry["baseline"]["path"],
                    "optimized_paths": ";".join(
                        opt["path"] for opt in entry["optimized"]
                    ),
                    "severity": entry["severity"],
                    "mismatch_count": len(entry["mismatches"]),
                    "mismatches": " | ".join(entry["mismatches"]),
                }
            )


def write_markdown(report: List[Dict[str, object]], path: Path) -> None:
    total = len(report)
    mismatched = sum(1 for entry in report if entry["mismatches"])
    ok = total - mismatched
    buckets = {
        "clean": sum(1 for entry in report if entry["severity"] == "clean"),
        "low": sum(1 for entry in report if entry["severity"] == "low"),
        "medium": sum(1 for entry in report if entry["severity"] == "medium"),
        "high": sum(1 for entry in report if entry["severity"] == "high"),
    }
    lines = [
        "# Benchmark Alignment Report",
        "",
        f"- Total pairs: {total}",
        f"- Clean pairs: {ok}",
        f"- Pairs with mismatches: {mismatched}",
        f"- Severity buckets: clean={buckets['clean']}, low={buckets['low']}, medium={buckets['medium']}, high={buckets['high']}",
        "",
        "| Chapter | Example | Severity | Mismatches |",
        "| --- | --- | --- | --- |",
    ]
    for entry in report:
        mismatches = entry["mismatches"]
        if not mismatches:
            continue
        bullet = "<br>".join(mismatches)
        lines.append(f"| {entry['chapter']} | {entry['example']} | {entry['severity']} | {bullet} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check baseline/optimized alignment")
    parser.add_argument("--json", type=Path, help="Path to write JSON report")
    parser.add_argument("--csv", type=Path, help="Path to write CSV summary")
    parser.add_argument("--markdown", type=Path, help="Path to write Markdown summary")
    parser.add_argument(
        "--chapter",
        action="append",
        default=[],
        help="Filter to specific chapter(s) (e.g., ch13 or 13). Can be repeated.",
    )
    args = parser.parse_args()

    chapter_filters = normalize_chapter_filters(args.chapter)
    chapters = [
        ch for ch in discover_all_chapters(repo_root)
        if not chapter_filters or ch.name in chapter_filters
    ]
    report: List[Dict[str, object]] = []

    for chapter_dir in chapters:
        pairs = discover_benchmarks(chapter_dir)
        if not pairs:
            continue
        for baseline_path, optimized_paths, example_name in pairs:
            baseline_meta = extract_metadata(baseline_path)
            optimized_meta = [extract_metadata(p) for p in optimized_paths]
            comparison = compare_pair(baseline_meta, optimized_meta)
            comparison["chapter"] = chapter_dir.name
            comparison["example"] = example_name
            report.append(comparison)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(report, args.csv)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.markdown)


def normalize_chapter_filters(raw_filters: List[str]) -> Set[str]:
    normalized: Set[str] = set()
    for entry in raw_filters:
        if not entry:
            continue
        tokens = entry.replace(",", " ").split()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if not token.startswith("ch"):
                token = f"ch{token}"
            normalized.add(token)
    return normalized


if __name__ == "__main__":
    main()
