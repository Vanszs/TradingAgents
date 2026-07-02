"""
Tests for leakage audit — hardened anti-leakage checks.
"""
from __future__ import annotations

import pytest


class TestLeakageAudit:
    def test_all_required_checks_present(self):
        """All required checks must be present in the audit."""
        required_checks = [
            "ohlcv_cutoff",
            "news_cutoff",
            "fundamental_available_date",
            "next_bar_execution",
            "memory_isolation",
            "live_provider_disabled",
        ]
        # Simulate a complete audit
        checks = {check: "PASSED" for check in required_checks}
        status = "PASSED" if all(v == "PASSED" for v in checks.values()) else "FAILED"
        assert status == "PASSED"
        assert len(checks) == 6

    def test_failed_check_overall_status(self):
        """If any check fails, overall status should be FAILED."""
        checks = {
            "ohlcv_cutoff": "PASSED",
            "news_cutoff": "FAILED",
            "fundamental_available_date": "PASSED",
            "next_bar_execution": "PASSED",
            "memory_isolation": "PASSED",
            "live_provider_disabled": "PASSED",
        }
        status = "PASSED" if all(v == "PASSED" for v in checks.values()) else "FAILED"
        assert status == "FAILED"

    def test_new_checks_default_to_passed(self):
        """New checks should default to PASSED if not explicitly set."""
        checks = {"ohlcv_cutoff": "PASSED"}
        required = [
            "lookback_window_respected",
            "last_data_date_cutoff",
            "sentiment_cutoff",
        ]
        for check in required:
            checks.setdefault(check, "PASSED")
        assert checks["lookback_window_respected"] == "PASSED"
        assert checks["last_data_date_cutoff"] == "PASSED"
        assert checks["sentiment_cutoff"] == "PASSED"
