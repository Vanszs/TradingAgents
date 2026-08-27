"""Aggressive legacy mode is still spot BUY/WNS only."""
from tradingagents.backtesting.decision_state_manager import DecisionStateManager
from tradingagents.backtesting.position import DecisionMappingConfig, ExtendedDecision, Position


def _decision(rating: str) -> ExtendedDecision:
    return ExtendedDecision(
        decision_id="T-1", ticker="TEST", trade_date="2026-01-01",
        agent_rating=rating, normalized_rating=rating.upper(),
    )


def test_aggressive_buy_flat_opens_long():
    result = DecisionStateManager(
        DecisionMappingConfig(mode="aggressive", allow_reverse_on_buy_sell=True)
    ).map(_decision("Buy"), Position(ticker="TEST"))
    assert result.position_intent == "open"
    assert result.target_position_side == "LONG"
    assert result.futures_action == "BUY_TO_OPEN"


def test_aggressive_wns_never_reverses():
    result = DecisionStateManager(
        DecisionMappingConfig(mode="aggressive", allow_reverse_on_buy_sell=True)
    ).map(_decision("Sell"), Position(ticker="TEST", quantity=100))
    assert result.position_intent == "hold"
    assert result.futures_action == "NO_ORDER"
    assert result.target_position_side == "LONG"


def test_aggressive_rejects_short_flags():
    try:
        DecisionMappingConfig(mode="aggressive", allow_short_on_sell=True)
    except ValueError:
        pass
    else:
        raise AssertionError("short mode must be rejected")
