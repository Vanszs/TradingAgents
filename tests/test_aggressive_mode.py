"""
Tests for aggressive mode decision mapping — PRD §10.3.
"""
from __future__ import annotations

import pytest

from tradingagents.backtesting.decision_state_manager import DecisionStateManager
from tradingagents.backtesting.position import (
    DecisionMappingConfig,
    ExtendedDecision,
    Position,
    PositionIntent,
    PositionSide,
)


class TestAggressiveMode:
    def setup_method(self):
        self.config = DecisionMappingConfig(mode="aggressive", allow_reverse_on_buy_sell=True)
        self.dsm = DecisionStateManager(config=self.config)

    def test_flat_strong_buy_opens(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Buy",
            normalized_rating="BUY",
            allocation_pct=20.0,
        )
        position = Position(ticker="TEST", quantity=0)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.OPEN.value
        assert result.target_position_side == PositionSide.LONG.value

    def test_long_buy_increases(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Buy",
            normalized_rating="BUY",
            allocation_pct=10.0,
        )
        position = Position(ticker="TEST", quantity=100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.INCREASE.value

    def test_long_strong_sell_reverse(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Strong_sell",
            normalized_rating="STRONG_SELL",
        )
        position = Position(ticker="TEST", quantity=100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.REVERSE.value
        assert result.target_position_side == PositionSide.SHORT.value

    def test_long_strong_sell_close_no_reverse(self):
        config = DecisionMappingConfig(mode="aggressive", allow_reverse_on_buy_sell=False)
        dsm = DecisionStateManager(config=config)
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Strong_sell",
            normalized_rating="STRONG_SELL",
        )
        position = Position(ticker="TEST", quantity=100)
        result = dsm.map(decision, position)
        assert result.position_intent == PositionIntent.CLOSE.value
        assert result.target_position_side == PositionSide.FLAT.value

    def test_short_sell_increases(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Sell",
            normalized_rating="SELL",
            allocation_pct=10.0,
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.INCREASE.value

    def test_short_strong_buy_reverse(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Strong_buy",
            normalized_rating="STRONG_BUY",
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.REVERSE.value
        assert result.target_position_side == PositionSide.LONG.value
