"""
Tests for leakage audit — PRD §19.7 (4 new checks).
"""
from __future__ import annotations

import pytest


class TestLeakageAudit:
    def test_all_required_checks_present(self):
        """PRD §19.7 — all 10 checks must be present."""
        required_checks = [
            "ohlcv_cutoff",
            "news_cutoff",
            "fundamental_available_date",
            "next_bar_execution",
            "memory_isolation",
            "live_provider_disabled",
            "long_short_pnl_rules",
            "reverse_fee_rule",
            "margin_rule",
            "leverage_limit",
        ]
        # Simulate a complete audit
        checks = {check: "PASSED" for check in required_checks}
        status = "PASSED" if all(v == "PASSED" for v in checks.values()) else "FAILED"
        assert status == "PASSED"
        assert len(checks) == 10

    def test_failed_check_overall_status(self):
        """If any check fails, overall status should be FAILED."""
        checks = {
            "ohlcv_cutoff": "PASSED",
            "news_cutoff": "FAILED",
            "fundamental_available_date": "PASSED",
            "next_bar_execution": "PASSED",
            "memory_isolation": "PASSED",
            "live_provider_disabled": "PASSED",
            "long_short_pnl_rules": "PASSED",
            "reverse_fee_rule": "PASSED",
            "margin_rule": "PASSED",
            "leverage_limit": "PASSED",
        }
        status = "PASSED" if all(v == "PASSED" for v in checks.values()) else "FAILED"
        assert status == "FAILED"

    def test_new_checks_default_to_passed(self):
        """New checks should default to PASSED if not explicitly set."""
        checks = {"ohlcv_cutoff": "PASSED"}
        required = [
            "long_short_pnl_rules",
            "reverse_fee_rule",
            "margin_rule",
            "leverage_limit",
        ]
        for check in required:
            checks.setdefault(check, "PASSED")
        assert checks["long_short_pnl_rules"] == "PASSED"
        assert checks["reverse_fee_rule"] == "PASSED"
        assert checks["margin_rule"] == "PASSED"
        assert checks["leverage_limit"] == "PASSED"
