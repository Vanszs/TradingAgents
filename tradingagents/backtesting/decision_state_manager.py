"""Map agent ratings to spot long-only backtest orders."""
from __future__ import annotations

from typing import Optional

from .position import (
    DecisionMappingConfig,
    ExtendedDecision,
    OrderType,
    Position,
    PositionIntent,
)

RATING_BUY = "Buy"
RATING_WNS = "WNS"
# Legacy import aliases; all normalize to the two canonical outputs.
RATING_OVERWEIGHT = RATING_BUY
RATING_HOLD = RATING_WNS
RATING_UNDERWEIGHT = RATING_WNS
RATING_SELL = RATING_WNS


def _canonical_rating(value: Optional[str]) -> str:
    """Normalize legacy labels at the backtest input boundary."""
    if value is None:
        return RATING_WNS
    clean = str(value).strip().lower().replace("_", " ")
    return RATING_BUY if clean in {"buy", "overweight", "strong buy", "strong_buy"} else RATING_WNS


class DecisionStateManager:
    """Pure BUY/WNS to long-only execution mapper.

    Existing positions never create agent entries. Static risk controls own
    every SELL_TO_CLOSE exit; this mapper emits only BUY_TO_OPEN or NO_ORDER.
    """

    def __init__(self, config: DecisionMappingConfig):
        self.config = config

    def resolve(
        self,
        rating: Optional[str],
        current_position: Position,
        allow_new_position: bool = True,
    ) -> tuple[PositionIntent, str, OrderType]:
        canonical = _canonical_rating(rating)
        side = current_position.side.value
        if canonical == RATING_WNS:
            return PositionIntent.HOLD, side, OrderType.NO_ORDER
        if side == "FLAT":
            if not allow_new_position:
                return PositionIntent.HOLD, "FLAT", OrderType.NO_ORDER
            return PositionIntent.OPEN, "LONG", OrderType.BUY_TO_OPEN
        if side == "LONG":
            # One-shot spot entries: BUY while already long is WNS/no-order.
            # Position size never pyramids from repeated daily ratings.
            return PositionIntent.HOLD, "LONG", OrderType.NO_ORDER
        # Legacy short state is not a valid backtest state. Do not cover or
        # reverse it from the canonical mapper.
        return PositionIntent.HOLD, side, OrderType.NO_ORDER

    def map(self, decision: ExtendedDecision, current_position: Position) -> ExtendedDecision:
        if not decision.valid:
            return ExtendedDecision.invalid(
                ticker=decision.ticker,
                trade_date=decision.trade_date,
                reason=decision.invalid_reason or "Invalid decision",
                source_report_path=decision.source_report_path,
            )

        rating = _canonical_rating(decision.agent_rating)
        intent, target_side, order_type = self.resolve(
            rating, current_position, allow_new_position=decision.allow_new_position
        )
        allocation = self.config.initial_entry_pct if intent == PositionIntent.OPEN else 0.0

        return ExtendedDecision(
            decision_id=decision.decision_id,
            ticker=decision.ticker,
            trade_date=decision.trade_date,
            report_generated_at=decision.report_generated_at,
            last_data_date=decision.last_data_date,
            decision_valid_from=decision.decision_valid_from,
            agent_rating=rating,
            normalized_rating=rating.upper(),
            allocation_pct=allocation,
            reduce_pct=decision.reduce_pct,
            leverage=1.0,
            confidence=decision.confidence,
            short_allowed=False,
            allow_new_position=decision.allow_new_position,
            market_mode="SPOT_LONG_ONLY",
            allowed_position_sides="LONG",
            position_intent=intent.value,
            current_position_side=current_position.side.value,
            target_position_side=target_side,
            futures_action=order_type.value,
            stop_price=decision.stop_price,
            take_profit=decision.take_profit,
            planned_entry_price=decision.planned_entry_price,
            wns_trigger_price=decision.wns_trigger_price,
            wns_recheck_date=decision.wns_recheck_date,
            next_review_date=decision.next_review_date,
            time_horizon_days=decision.time_horizon_days,
            time_horizon_label=decision.time_horizon_label,
            thesis_summary=decision.thesis_summary,
            source_report_path=decision.source_report_path,
            raw_text_excerpt=decision.raw_text_excerpt,
            valid=True,
        )
