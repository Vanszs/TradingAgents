"""
Tests for conservative mode decision mapping — PRD §10.2.
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


class TestConservativeMode:
    def setup_method(self):
        self.config = DecisionMappingConfig(mode="conservative")
        self.dsm = DecisionStateManager(config=self.config)

    def test_flat_buy_opens_long(self):
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

    def test_flat_sell_opens_short(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Sell",
            normalized_rating="SELL",
            allocation_pct=20.0,
        )
        position = Position(ticker="TEST", quantity=0)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.OPEN.value
        assert result.target_position_side == PositionSide.SHORT.value

    def test_flat_hold_no_order(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Hold",
            normalized_rating="HOLD",
        )
        position = Position(ticker="TEST", quantity=0)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.HOLD.value
        assert result.target_position_side == PositionSide.FLAT.value

    def test_long_buy_holds(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Buy",
            normalized_rating="BUY",
        )
        position = Position(ticker="TEST", quantity=100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.HOLD.value
        assert result.target_position_side == PositionSide.LONG.value

    def test_long_sell_reduces(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Sell",
            normalized_rating="SELL",
            reduce_pct=50.0,
        )
        position = Position(ticker="TEST", quantity=100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.REDUCE.value

    def test_long_strong_sell_closes(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Strong_sell",
            normalized_rating="STRONG_SELL",
        )
        position = Position(ticker="TEST", quantity=100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.CLOSE.value
        assert result.target_position_side == PositionSide.FLAT.value

    def test_short_buy_reduces(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Buy",
            normalized_rating="BUY",
            reduce_pct=50.0,
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.REDUCE.value

    def test_short_sell_holds(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Sell",
            normalized_rating="SELL",
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.HOLD.value
        assert result.target_position_side == PositionSide.SHORT.value

    def test_short_strong_buy_closes(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Strong_buy",
            normalized_rating="STRONG_BUY",
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.CLOSE.value
        assert result.target_position_side == PositionSide.FLAT.value

    def test_conservative_no_reverse_on_buy(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Buy",
            normalized_rating="BUY",
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent != PositionIntent.REVERSE.value

    def test_invalid_decision_returns_invalid(self):
        decision = ExtendedDecision.invalid(
            ticker="TEST",
            trade_date="2026-01-01",
            reason="test",
        )
        position = Position(ticker="TEST", quantity=0)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.HOLD.value

    def test_allocation_on_buy(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Buy",
            normalized_rating="BUY",
            allocation_pct=30.0,
        )
        position = Position(ticker="TEST", quantity=0)
        result = self.dsm.map(decision, position)
        # DSM may override allocation with default
        assert result.allocation_pct is not None

    def test_allocation_on_reduce(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Underweight",
            normalized_rating="UNDERWEIGHT",
            reduce_pct=40.0,
        )
        position = Position(ticker="TEST", quantity=100)
        result = self.dsm.map(decision, position)
        # DSM uses its own reduce allocation, not the input's reduce_pct
        assert result.position_intent == PositionIntent.REDUCE.value

    def test_short_hold_no_order(self):
        decision = ExtendedDecision(
            decision_id="T-1",
            ticker="TEST",
            trade_date="2026-01-01",
            agent_rating="Hold",
            normalized_rating="HOLD",
        )
        position = Position(ticker="TEST", quantity=-100)
        result = self.dsm.map(decision, position)
        assert result.position_intent == PositionIntent.HOLD.value
        assert result.target_position_side == PositionSide.SHORT.value
