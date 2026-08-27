"""Legacy mode names now share the spot BUY/WNS mapper."""
from tradingagents.backtesting.decision_state_manager import DecisionStateManager
from tradingagents.backtesting.position import DecisionMappingConfig, ExtendedDecision, Position


def _decision(rating: str, allow_new_position: bool = True) -> ExtendedDecision:
    return ExtendedDecision(
        decision_id="T-1", ticker="TEST", trade_date="2026-01-01",
        agent_rating=rating, normalized_rating=rating.upper(),
        allow_new_position=allow_new_position,
    )


def test_conservative_buy_flat_opens_long():
    result = DecisionStateManager(DecisionMappingConfig(mode="conservative")).map(
        _decision("Buy"), Position(ticker="TEST")
    )
    assert result.position_intent == "open"
    assert result.target_position_side == "LONG"
    assert result.market_mode == "SPOT_LONG_ONLY"


def test_conservative_wns_flat_has_no_order():
    result = DecisionStateManager(DecisionMappingConfig(mode="conservative")).map(
        _decision("Hold"), Position(ticker="TEST")
    )
    assert result.position_intent == "hold"
    assert result.futures_action == "NO_ORDER"


def test_conservative_buy_respects_no_new_position():
    result = DecisionStateManager(DecisionMappingConfig(mode="conservative")).map(
        _decision("Buy", allow_new_position=False), Position(ticker="TEST")
    )
    assert result.futures_action == "NO_ORDER"


def test_conservative_long_buy_does_not_pyramid():
    result = DecisionStateManager(DecisionMappingConfig(mode="conservative")).map(
        _decision("Buy"), Position(ticker="TEST", quantity=100)
    )
    assert result.futures_action == "NO_ORDER"
    assert result.position_intent == "hold"


def test_conservative_modes_cannot_enable_shorts():
    try:
        DecisionMappingConfig(mode="conservative", short_allowed=True)
    except ValueError:
        pass
    else:
        raise AssertionError("short mode must be rejected")
