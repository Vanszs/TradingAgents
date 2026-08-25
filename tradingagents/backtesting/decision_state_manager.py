"""
Decision state manager — 5-tier rating → futures-style action mapper.

Two modes are supported:

* ``mode="strict_5tier"`` (default) — Preserves the full 5-tier agent rating
  (Buy / Overweight / Hold / Underweight / Sell) and applies the state-aware
  table from the backtesting PRD spec.

  Mapping table (state-aware)::

    Rating \\ Pos | FLAT          | LONG           | SHORT
    --------------+---------------+----------------+-----------------
    Buy           | OPEN_LONG     | HOLD_LONG      | CLOSE_SHORT
    Overweight    | OPEN_LONG *   | HOLD_LONG      | CLOSE_SHORT
    Hold          | NO_ACTION     | HOLD_LONG      | HOLD_SHORT
    Underweight   | NO_ACTION     | REDUCE_LONG    | ADD_SHORT *
    Sell          | OPEN_SHORT    | CLOSE_LONG     | HOLD_SHORT

  ``*`` = sized down by the corresponding ``overweight_size_multiplier`` /
  ``underweight_size_multiplier``.

  In ``strict_5tier`` mode the mapper **never** issues a direct
  LONG-to-SHORT (or SHORT-to-LONG) reverse. The only way to flip sides is
  through an explicit two-leg ``REVERSE_TO_*`` order, gated by
  ``allow_explicit_reverse=True``.

* ``mode="legacy_prd"`` — Original 3-bucket behaviour from the PRD
  (Buy/Overweight → buy bias, Hold → hold, Underweight/Sell → sell bias).
  This is kept for backward compatibility with the existing test suite.

The class is intentionally pure: it does not look at OHLCV, news or any
live data source. Inputs are the parsed decision, the current position
state, and the configuration.
"""
from __future__ import annotations

from typing import Any, Optional

from .position import (
    DecisionMappingConfig,
    ExtendedDecision,
    OrderType,
    Position,
    PositionIntent,
)

# Canonical 5-tier rating vocabulary, in agent-facing form.
RATING_BUY = "Buy"
RATING_OVERWEIGHT = "Overweight"
RATING_HOLD = "Hold"
RATING_WNS = "WNS"
RATING_UNDERWEIGHT = "Underweight"
RATING_SELL = "Sell"

# Lower-cased canonical form used internally.
_RATING_CANONICAL = {
    "buy": RATING_BUY,
    "overweight": RATING_OVERWEIGHT,
    "hold": RATING_HOLD,
    "wns": RATING_WNS,
    "wait and see": RATING_WNS,
    "wait_and_see": RATING_WNS,
    "wait & see": RATING_WNS,
    "underweight": RATING_UNDERWEIGHT,
    "sell": RATING_SELL,
}

# Legacy 3-bucket normalisation (PRD original). Kept for ``legacy_prd`` mode
# and for any callers that still emit ``strong_buy`` / ``strong_sell``.
# Preserves strong_buy/strong_sell distinction for the legacy allocation
# heuristic (strong_buy → 30% allocation).
_RATING_LEGACY_NORMALIZE = {
    "strong_buy": "strong_buy",
    "buy": "buy",
    "overweight": "buy",
    "hold": "hold",
    "underweight": "sell",
    "sell": "sell",
    "strong_sell": "strong_sell",
}


def _canonical_rating(value: str) -> Optional[str]:
    """Return one of the 5 canonical rating strings (case-insensitive) or None."""
    if value is None:
        return None
    return _RATING_CANONICAL.get(str(value).strip().lower())


class DecisionStateManager:
    """
    State-aware decision mapper.

    Translates a raw :class:`position.ParsedDecision` (with ``agent_rating``)
    into an :class:`position.ExtendedDecision` (which is the same dataclass)
    that carries ``position_intent``, ``target_position_side`` and
    ``futures_action``.

    The mapping is governed by:

    * the agent's rating (5-tier)
    * the current position side (LONG / SHORT / FLAT)
    * the ``DecisionMappingConfig`` (mode, allow_explicit_reverse,
      multipliers, default_reduce_pct, …)
    """

    def __init__(self, config: DecisionMappingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def resolve(
        self,
        rating: Optional[str],
        current_position: Position,
    ) -> tuple[PositionIntent, str, OrderType]:
        """Resolve rating against current position side."""
        canonical = _canonical_rating(rating)
        return self._strict_resolve(canonical, current_position.side.value)

    def map(
        self,
        decision: ExtendedDecision,
        current_position: Position,
    ) -> ExtendedDecision:
        """Map a ParsedDecision into an ExtendedDecision with intent + action."""
        if not decision.valid:
            return ExtendedDecision.invalid(
                ticker=decision.ticker,
                trade_date=decision.trade_date,
                reason=decision.invalid_reason or "Invalid decision",
                source_report_path=decision.source_report_path,
            )

        mode = self.config.mode
        if mode == "strict_5tier":
            return self._map_strict(decision, current_position)

        # legacy_prd, conservative, or aggressive all use the legacy path
        return self._map_legacy(decision, current_position)

    # ------------------------------------------------------------------
    # Strict 5-tier mapping (default / new behaviour)
    # ------------------------------------------------------------------
    def _map_strict(
        self,
        decision: ExtendedDecision,
        current_position: Position,
    ) -> ExtendedDecision:
        rating = _canonical_rating(decision.agent_rating)
        current_side = current_position.side.value

        # Determine intent, target side and order type from the strict table.
        intent, target_side, order_type = self._strict_resolve(rating, current_side)

        # Determine allocation percentage for sizing.
        allocation_pct = self._strict_allocation(rating, intent, current_side)

        # Hold preserves the existing position. If the agent's allocation_pct
        # is missing or zero, we leave it alone; if it is set we still respect
        # the no-action semantics by reporting 0.0 allocation.
        if intent == PositionIntent.HOLD:
            allocation_pct = 0.0

        return ExtendedDecision(
            decision_id=decision.decision_id,
            ticker=decision.ticker,
            trade_date=decision.trade_date,
            report_generated_at=decision.report_generated_at,
            last_data_date=decision.last_data_date,
            decision_valid_from=decision.decision_valid_from,
            agent_rating=rating or decision.agent_rating,
            normalized_rating=(rating or "Hold").upper(),
            allocation_pct=allocation_pct,
            reduce_pct=decision.reduce_pct,
            leverage=decision.leverage or 1.0,
            confidence=decision.confidence,
            short_allowed=self._strict_short_allowed(rating),
            allow_new_position=decision.allow_new_position,
            market_mode="FUTURES_STYLE_SIMULATION",
            allowed_position_sides="LONG,SHORT" if self._strict_short_allowed(rating) else "LONG",
            position_intent=intent.value,
            current_position_side=current_side,
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

    def _strict_resolve(
        self,
        rating: Optional[str],
        current_side: str,
    ) -> tuple[PositionIntent, str, OrderType]:
        """Apply the strict 5-tier table (rating × current_side → intent + order)."""
        if rating in (RATING_HOLD, RATING_WNS, None):
            return PositionIntent.HOLD, current_side, OrderType.NO_ORDER

        if rating == RATING_BUY:
            if current_side == "FLAT":
                return PositionIntent.OPEN, "LONG", OrderType.BUY_TO_OPEN
            if current_side == "LONG":
                return PositionIntent.HOLD, "LONG", OrderType.NO_ORDER
            if current_side == "SHORT":
                # Conservative: close the short, do NOT auto-reverse.
                return PositionIntent.CLOSE, "FLAT", OrderType.BUY_TO_CLOSE
            return PositionIntent.HOLD, current_side, OrderType.NO_ORDER

        if rating == RATING_OVERWEIGHT:
            if current_side == "FLAT":
                return PositionIntent.OPEN, "LONG", OrderType.BUY_TO_OPEN
            if current_side == "LONG":
                # Pyramiding: add to long when rating aligned.
                return PositionIntent.INCREASE, "LONG", OrderType.BUY_TO_ADD
            if current_side == "SHORT":
                # Gradual exit: reduce short when rating opposite.
                return PositionIntent.REDUCE, "SHORT", OrderType.BUY_TO_REDUCE
            return PositionIntent.HOLD, current_side, OrderType.NO_ORDER

        allow_short = bool(
            getattr(self.config, "allow_short_on_sell", False)
            and getattr(self.config, "short_allowed", False)
        )
        allow_short_underweight = bool(
            getattr(self.config, "allow_short_on_underweight", False)
            and getattr(self.config, "short_allowed", False)
        )

        if rating == RATING_UNDERWEIGHT:
            if current_side == "LONG":
                # Gradual exit: reduce long when rating opposite.
                return PositionIntent.REDUCE, "LONG", OrderType.SELL_TO_REDUCE
            if current_side == "FLAT":
                if allow_short_underweight:
                    return PositionIntent.OPEN, "SHORT", OrderType.SELL_TO_OPEN
                return PositionIntent.HOLD, "FLAT", OrderType.NO_ORDER
            if current_side == "SHORT":
                # Pyramiding: add to short when rating aligned.
                return PositionIntent.INCREASE, "SHORT", OrderType.SELL_TO_ADD
            return PositionIntent.HOLD, current_side, OrderType.NO_ORDER

        if rating == RATING_SELL:
            if current_side == "LONG":
                # Conservative: close the long, do NOT auto-reverse.
                return PositionIntent.CLOSE, "FLAT", OrderType.SELL_TO_CLOSE
            if current_side == "FLAT":
                if allow_short:
                    return PositionIntent.OPEN, "SHORT", OrderType.SELL_TO_OPEN
                return PositionIntent.HOLD, "FLAT", OrderType.NO_ORDER
            if current_side == "SHORT":
                return PositionIntent.HOLD, "SHORT", OrderType.NO_ORDER
            return PositionIntent.HOLD, current_side, OrderType.NO_ORDER

        # Unknown rating → no action.
        return PositionIntent.HOLD, current_side, OrderType.NO_ORDER

    def _strict_allocation(
        self,
        rating: Optional[str],
        intent: PositionIntent,
        current_side: str,
    ) -> float:
        """Pick the allocation percentage for the strict table."""
        if intent == PositionIntent.CLOSE:
            return 1.0  # 100% close
        if intent == PositionIntent.REDUCE:
            return self.config.reduce_step_pct  # 10% gradual exit
        if intent == PositionIntent.INCREASE:
            return self.config.pyramid_pct  # 10% pyramiding

        if intent == PositionIntent.OPEN:
            return self.config.initial_entry_pct  # 30% first entry
        return 0.0

    def _strict_short_allowed(self, rating: Optional[str]) -> bool:
        """Whether the active config permits opening a new short."""
        if rating == RATING_SELL:
            return self.config.allow_short_on_sell
        if rating == RATING_UNDERWEIGHT:
            return self.config.allow_short_on_underweight
        return False

    # ------------------------------------------------------------------
    # Legacy 3-bucket mapping (kept for backward-compat tests)
    # ------------------------------------------------------------------
    def _map_legacy(
        self,
        decision: ExtendedDecision,
        current_position: Position,
    ) -> ExtendedDecision:
        rating = _RATING_LEGACY_NORMALIZE.get(
            (decision.agent_rating or "").lower(), "hold"
        )
        current_side = current_position.side.value

        # Determine effective submode. ``aggressive`` and ``conservative``
        # are top-level mode aliases; ``legacy_prd`` uses ``legacy_submode``.
        if self.config.mode == "aggressive":
            aggressive = True
        elif self.config.mode == "conservative":
            aggressive = False
        else:  # legacy_prd
            aggressive = self.config.legacy_submode == "aggressive"

        if current_side == "FLAT":
            if rating in ("buy", "strong_buy"):
                intent = PositionIntent.OPEN
                target = "LONG"
                order_type = OrderType.BUY_TO_OPEN
            elif rating in ("sell", "strong_sell") and self.config.allow_short_on_sell:
                intent = PositionIntent.OPEN
                target = "SHORT"
                order_type = OrderType.SELL_TO_OPEN
            else:
                intent = PositionIntent.HOLD
                target = "FLAT"
                order_type = OrderType.NO_ORDER
        elif current_side == "LONG":
            if rating in ("buy", "strong_buy"):
                if aggressive:
                    intent = PositionIntent.INCREASE
                    order_type = OrderType.BUY_TO_ADD
                else:
                    intent = PositionIntent.HOLD
                    order_type = OrderType.NO_ORDER
                target = "LONG"
            elif rating == "hold":
                intent = PositionIntent.HOLD
                target = "LONG"
                order_type = OrderType.NO_ORDER
            elif rating == "sell":
                # Legacy conservative: "sell" rating reduces long, doesn't
                # close it outright.
                intent = PositionIntent.REDUCE
                target = "LONG"
                order_type = OrderType.SELL_TO_REDUCE
            elif rating == "strong_sell":
                if self.config.allow_reverse_on_buy_sell and self.config.allow_short_on_sell:
                    intent = PositionIntent.REVERSE
                    target = "SHORT"
                    order_type = OrderType.REVERSE_TO_SHORT
                else:
                    intent = PositionIntent.CLOSE
                    target = "FLAT"
                    order_type = OrderType.SELL_TO_CLOSE
            else:
                intent = PositionIntent.HOLD
                target = "LONG"
                order_type = OrderType.NO_ORDER
        else:  # SHORT
            if rating in ("sell", "strong_sell"):
                if aggressive:
                    intent = PositionIntent.INCREASE
                    order_type = OrderType.SELL_TO_ADD
                else:
                    intent = PositionIntent.HOLD
                    order_type = OrderType.NO_ORDER
                target = "SHORT"
            elif rating == "hold":
                intent = PositionIntent.HOLD
                target = "SHORT"
                order_type = OrderType.NO_ORDER
            elif rating == "buy":
                # Legacy conservative: "buy" rating reduces short, doesn't
                # close it outright.
                intent = PositionIntent.REDUCE
                target = "SHORT"
                order_type = OrderType.BUY_TO_REDUCE
            elif rating == "strong_buy":
                if self.config.allow_reverse_on_buy_sell:
                    intent = PositionIntent.REVERSE
                    target = "LONG"
                    order_type = OrderType.REVERSE_TO_LONG
                else:
                    intent = PositionIntent.CLOSE
                    target = "FLAT"
                    order_type = OrderType.BUY_TO_CLOSE
            else:
                intent = PositionIntent.HOLD
                target = "SHORT"
                order_type = OrderType.NO_ORDER

        allocation_pct = self._legacy_allocation(rating, intent)

        return ExtendedDecision(
            decision_id=decision.decision_id,
            ticker=decision.ticker,
            trade_date=decision.trade_date,
            report_generated_at=decision.report_generated_at,
            last_data_date=decision.last_data_date,
            decision_valid_from=decision.decision_valid_from,
            agent_rating=decision.agent_rating,
            normalized_rating=rating.upper() if rating else "HOLD",
            allocation_pct=allocation_pct,
            reduce_pct=decision.reduce_pct,
            leverage=decision.leverage or 1.0,
            confidence=decision.confidence,
            short_allowed=self._legacy_short_allowed(rating),
            allow_new_position=decision.allow_new_position,
            market_mode="FUTURES_STYLE_SIMULATION",
            allowed_position_sides=("LONG,SHORT" if self._legacy_short_allowed(rating) else "LONG"),
            position_intent=intent.value,
            current_position_side=current_side,
            target_position_side=target,
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

    def _legacy_allocation(self, rating: str, intent: PositionIntent) -> float:
        if intent in (PositionIntent.CLOSE, PositionIntent.REDUCE):
            return self.config.default_reduce_pct
        if intent == PositionIntent.INCREASE:
            return self.config.overweight_size_multiplier
        if intent == PositionIntent.OPEN:
            if rating == "strong_buy":
                return 0.30
            if rating == "buy":
                return 0.20
            if rating in ("sell", "strong_sell"):
                return 0.20
            return 0.25
        return 0.0

    def _legacy_short_allowed(self, rating: str) -> bool:
        if rating in ("sell", "strong_sell"):
            return self.config.allow_short_on_sell
        return False
