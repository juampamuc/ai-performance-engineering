from __future__ import annotations

from labs.ozaki_scheme.lab_utils import (
    format_result_row,
    parse_float_csv,
    parse_int_csv,
    parse_metrics,
    summarize_reproducibility,
)


def test_ozaki_lab_parse_metrics_captures_strategy_and_checksum() -> None:
    stdout = """
VARIANT: ozaki_dynamic
EMULATION_STRATEGY: eager
EMULATION_USED: 1
RETAINED_BITS: 4
TFLOPS: 11.178
MAX_ABS_ERROR: 3.0e-06
MEAN_ABS_ERROR: 0.0
RESULT_CHECKSUM: 1.2345000000e+02
TIME_MS: 12.296
"""

    metrics = parse_metrics(stdout)

    assert metrics["variant"] == "ozaki_dynamic"
    assert metrics["emulation_strategy"] == "eager"
    assert metrics["emulation_used"] == 1
    assert metrics["retained_bits"] == 4
    assert metrics["tflops"] == 11.178
    assert metrics["checksum"] == 123.45
    assert metrics["time_ms"] == 12.296


def test_ozaki_lab_csv_parsers_preserve_order() -> None:
    assert parse_int_csv("6,8,10,12") == [6, 8, 10, 12]
    assert parse_float_csv("1e-1,1e-2,1e-3") == [1e-1, 1e-2, 1e-3]


def test_ozaki_lab_reproducibility_summary_flags_stable_records() -> None:
    records = [
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
    ]

    summary = summarize_reproducibility(records)

    assert summary == {
        "run_count": 3,
        "checksum_stable": True,
        "retained_bits_stable": True,
        "emulation_used_stable": True,
    }


def test_ozaki_lab_reproducibility_summary_flags_drift() -> None:
    records = [
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
        {"checksum": 10.1, "retained_bits": 4, "emulation_used": 1},
    ]

    summary = summarize_reproducibility(records)

    assert summary["checksum_stable"] is False
    assert summary["retained_bits_stable"] is True
    assert summary["emulation_used_stable"] is True


def test_ozaki_lab_result_row_reports_speedup() -> None:
    row = format_result_row(
        "Ozaki dynamic",
        {
            "time_ms": 2.0,
            "tflops": 100.0,
            "retained_bits": 4,
            "emulation_used": 1,
            "max_abs_error": 1e-6,
            "mean_abs_error": 0.0,
        },
        baseline_ms=10.0,
    )

    assert "| Ozaki dynamic | 2.000 | 100.000 | 5.00x | 4 | 1 | 1.000e-06 | 0.000e+00 |" == row
