from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from run_lab import _compute_mode_metrics, _detect_injection_success, _selected_modes


def test_selected_modes() -> None:
    assert _selected_modes("both") == ["naive", "contract"]
    assert _selected_modes("naive") == ["naive"]
    assert _selected_modes("contract") == ["contract"]


def test_compute_mode_metrics() -> None:
    rows = [
        {
            "is_answerable": True,
            "is_injection_case": False,
            "citation_coverage_ok": True,
            "refusal_correct": True,
            "injection_succeeded": False,
            "format_ok": False,
        },
        {
            "is_answerable": False,
            "is_injection_case": False,
            "citation_coverage_ok": True,
            "refusal_correct": True,
            "injection_succeeded": False,
            "format_ok": True,
        },
        {
            "is_answerable": False,
            "is_injection_case": True,
            "citation_coverage_ok": True,
            "refusal_correct": False,
            "injection_succeeded": True,
            "format_ok": True,
        },
    ]

    metrics = _compute_mode_metrics(rows)
    assert metrics["query_count"] == 3
    assert metrics["citation_coverage_rate"] == 1.0
    assert metrics["refusal_correctness_rate"] == 0.5
    assert metrics["injection_success_rate"] == 1.0
    assert metrics["format_ok_rate"] == 2 / 3


def test_injection_detector() -> None:
    assert _detect_injection_success("system override") is True
    assert _detect_injection_success("normal answer") is False
