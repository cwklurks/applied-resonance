"""Lightweight tests for engine.smoke — arg parsing + report verification only.

These never execute the pipeline; the real end-to-end run is the smoke command
itself (`uv run python -m engine.smoke`).
"""

import pytest

from engine.smoke import parse_args, verify_report


def test_parse_args_defaults():
    args = parse_args([])
    assert args.subset == 50
    assert args.device == "cpu"
    assert args.budget_s == 600.0
    assert args.seed == 1337


def test_parse_args_overrides():
    args = parse_args(["--subset", "20", "--device", "cuda", "--budget-s", "120"])
    assert args.subset == 20
    assert args.device == "cuda"
    assert args.budget_s == 120.0


GOOD_REPORT = """# EarSight Baseline Report
## Anomaly detection — AUC / pAUC (p=0.1)
| **fan** | **type average** | | | **0.749** |
## Determinism
PASS — round-3 AUC/pAUC identical across two runs.
"""


def test_verify_report_passes_on_good_report():
    verify_report(GOOD_REPORT)  # must not raise


def test_verify_report_fails_when_determinism_not_pass():
    bad = GOOD_REPORT.replace("PASS — round-3", "FAIL — round-3")
    with pytest.raises(AssertionError):
        verify_report(bad)
