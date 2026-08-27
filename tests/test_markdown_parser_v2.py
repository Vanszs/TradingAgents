"""
Tests for markdown_parser.py — PRD §9 (Phase 4 version).
"""
from __future__ import annotations

import pytest

from tradingagents.backtesting.markdown_parser import MarkdownDecisionParser

SAMPLE_REPORT = """\
# Trading Analysis Report: BUMI.JK

Generated: 2026-05-25 16:30:00
Trade Date: 2026-05-25
Last Data Date: 2026-05-25
Decision Valid From: 2026-05-26

## Portfolio Manager Decision

Based on technical and fundamental analysis, the recommendation is:

Rating: Buy
Confidence: 72
Allocation: 20%
Leverage: 2
Stop Loss: 170
Take Profit: 210
Time Horizon Days: 20
Short Allowed: true
"""

SAMPLE_REPORT_SHORT_BLOCKED = """\
# Trading Analysis Report: TEST.JK

Generated: 2026-01-01 16:30:00
Trade Date: 2026-01-01
Last Data Date: 2026-01-01
Decision Valid From: 2026-01-02

## Trading Plan

Rating: Hold
Jangan buka posisi short.
"""

SAMPLE_REPORT_SELL = """\
# Trading Analysis Report: ACME

Generated: 2026-03-10 16:30:00
Trade Date: 2026-03-10
Last Data Date: 2026-03-10
Decision Valid From: 2026-03-11

## Final Decision

Rating: Sell
Reduce: 50%
Target: 80
"""


class TestMarkdownParserPhase4:
    def setup_method(self):
        self.parser = MarkdownDecisionParser()

    def test_extracts_rating(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.agent_rating == "Buy"
        assert d.normalized_rating == "BUY"

    def test_extracts_allocation(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.allocation_pct == 20.0

    def test_extracts_leverage(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.leverage == 2

    def test_extracts_stop_price(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.stop_price == 170.0

    def test_extracts_take_profit(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.take_profit == 210.0

    def test_extracts_time_horizon(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.time_horizon_days == 20

    def test_no_action_inference(self):
        """PRD §9: parser should NOT infer futures_action from text."""
        d = self.parser.parse_text(SAMPLE_REPORT)
        # The parser sets futures_action to default "NO_ORDER", never from text
        assert d.futures_action == "NO_ORDER"

    def test_short_allowed_is_always_false(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.short_allowed is False

    def test_short_allowed_false(self):
        d = self.parser.parse_text(SAMPLE_REPORT_SHORT_BLOCKED)
        assert d.short_allowed is False

    def test_sell_rating_normalizes_to_wns(self):
        d = self.parser.parse_text(SAMPLE_REPORT_SELL)
        assert d.agent_rating == "WNS"
        assert d.normalized_rating == "WNS"
        assert d.reduce_pct == 50.0
        assert d.take_profit == 80.0

    def test_invalid_missing_ticker(self):
        d = self.parser.parse_text("No ticker here. Rating: Buy")
        assert d.valid is False

    def test_invalid_missing_rating(self):
        d = self.parser.parse_text("Ticker: TEST\nNo rating here")
        assert d.valid is False

    def test_metadata_fields(self):
        d = self.parser.parse_text(SAMPLE_REPORT)
        assert d.ticker == "BUMI.JK"
        assert d.trade_date == "2026-05-25"
        assert d.decision_valid_from == "2026-05-26"

    def test_extracts_wns_and_price_triggers(self):
        wns_report = """\
# Trading Analysis Report: BBRI.JK

Generated: 2026-06-01 16:30:00
Trade Date: 2026-06-01
Decision Valid From: 2026-06-02

## Portfolio Manager Decision

Rating: Wait and See
Planned Entry Price: 4200.0
WNS Trigger Price: 4150.0
WNS Recheck Date: 2026-06-10
Time Horizon Days: 15
"""
        d = self.parser.parse_text(wns_report)
        assert d.agent_rating == "WNS"
        assert d.normalized_rating == "WNS"
        assert d.planned_entry_price == 4200.0
        assert d.wns_trigger_price == 4150.0
        assert d.wns_recheck_date == "2026-06-10"
        assert d.time_horizon_days == 15
