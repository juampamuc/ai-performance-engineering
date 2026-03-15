"""Warning visibility helpers for noisy runtime imports and optional components."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
import warnings
from typing import Iterator, Sequence

_CUDA_CAPABILITY_WARNING_PATTERNS: tuple[str, ...] = (
    ".*Found GPU.*cuda capability.*",
    ".*Found GPU.*which is of cuda capability.*",
    ".*Minimum and Maximum cuda capability supported.*",
)

_BENCHMARK_IMPORT_WARNING_PATTERNS: tuple[str, ...] = (
    ".*Please use the new API settings to control TF32.*",
    ".*TensorFloat32 tensor cores.*available but not enabled.*",
    ".*Overriding a previously registered kernel.*",
    ".*Warning only once for all operators.*",
)


def _reemit_warning(record: warnings.WarningMessage) -> None:
    warnings.warn_explicit(
        message=record.message,
        category=record.category,
        filename=record.filename,
        lineno=record.lineno,
        module=record.filename,
        source=record.source,
    )


def _warning_matches_patterns(record: warnings.WarningMessage, patterns: Sequence[str]) -> bool:
    category = record.category
    if category is None or not issubclass(category, UserWarning):
        return False
    message = str(record.message)
    return any(re.search(pattern, message) for pattern in patterns)


def _emit_captured_warning_summary(
    matched: Sequence[warnings.WarningMessage],
    *,
    context: str | None,
) -> None:
    if not matched:
        return
    if os.environ.get("AISP_SHOW_ALL_CAPTURED_WARNINGS", "0") == "1":
        for record in matched:
            warnings.warn(
                f"Captured known runtime warning during {context or 'runtime operation'}: "
                f"{record.category.__name__}: {record.message}",
                RuntimeWarning,
                stacklevel=3,
            )
        return

    unique_messages: list[str] = []
    seen = set()
    for record in matched:
        text = f"{record.category.__name__}: {record.message}"
        if text in seen:
            continue
        seen.add(text)
        unique_messages.append(text)
    examples = "; ".join(unique_messages[:3])
    if len(unique_messages) > 3:
        examples += f"; ... {len(unique_messages) - 3} more unique warning(s)"
    warnings.warn(
        f"Captured {len(matched)} known user warning(s) during {context or 'runtime operation'}. "
        "These warnings are no longer silently suppressed. "
        f"Examples: {examples}",
        RuntimeWarning,
        stacklevel=3,
    )


def warn_optional_component_unavailable(
    component: str,
    exc: BaseException,
    *,
    impact: str,
    context: str | None = None,
) -> None:
    location = f" during {context}" if context else ""
    warnings.warn(
        f"Optional component '{component}' is unavailable{location}: {exc}. Impact: {impact}",
        RuntimeWarning,
        stacklevel=2,
    )


@contextmanager
def suppress_user_warnings(
    patterns: Sequence[str],
    *,
    context: str | None = None,
) -> Iterator[None]:
    """Capture a narrow set of known-noisy user warnings and summarize them later."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    matched: list[warnings.WarningMessage] = []
    for record in caught:
        if _warning_matches_patterns(record, patterns):
            matched.append(record)
        else:
            _reemit_warning(record)
    _emit_captured_warning_summary(matched, context=context)


@contextmanager
def suppress_known_cuda_capability_warnings(*, context: str | None = None) -> Iterator[None]:
    """Capture PyTorch capability warnings only around targeted imports/probes."""
    with suppress_user_warnings(_CUDA_CAPABILITY_WARNING_PATTERNS, context=context):
        yield


@contextmanager
def suppress_benchmark_import_warnings(*, context: str | None = None) -> Iterator[None]:
    """Capture known import-time benchmark noise without mutating global filters."""
    with suppress_user_warnings(
        (*_CUDA_CAPABILITY_WARNING_PATTERNS, *_BENCHMARK_IMPORT_WARNING_PATTERNS),
        context=context,
    ):
        yield
