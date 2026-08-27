"""
Order generation for stock margin backtests — PRD-compliant version.

Translates an ExtendedDecision (or legacy ParsedDecision) into one or more
pending Orders with explicit OrderType (PRD §7.2):
- BUY_TO_OPEN, BUY_TO_ADD, SELL_TO_CLOSE, NO_ORDER

Supports:
- Lot-size rounding
- Max-leverage cap enforcement
- Percentage-based fees/slippage
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

import pandas as pd

from .position import (
    BacktestConfig,
    ExecutionConfig,
    ExtendedDecision,
    Fill,
    InstrumentSpec,
    MarketPoint,
    Order,
    OrderType,
    ParsedDecision,
    Position,
    PositionSide,
)

logger = logging.getLogger(__name__)


class OrderGenerator:
    """Generates pending Orders from ExtendedDecisions (PRD §7.2)."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exec_cfg = config.execution
        self.margin_cfg = config.margin
        self._dsm_config = config.decision_mapping

    # ==================================================================
    # New PRD-compliant entry point
    # ==================================================================
    def decide(
        self,
        decision: ExtendedDecision,
        current_position: Position,
        current_equity: float,
        current_cash: Optional[float] = None,
        reference_price: float = 0.0,
    ) -> list[Order]:
        """
        Map an ExtendedDecision to one or more Orders.
        Takes the decision (already mapped by DecisionStateManager) and produces orders.
        """
        if not decision.valid:
            return []
        order_type = OrderType(decision.futures_action)
        if order_type == OrderType.NO_ORDER:
            return []
        if order_type != OrderType.BUY_TO_OPEN:
            logger.error(
                "Rejected non-entry order %s for %s",
                order_type.value,
                decision.ticker,
            )
            return []
        if not current_position.is_flat():
            return []

        ticker = decision.ticker
        exec_date = decision.decision_valid_from
        if not exec_date or exec_date == decision.trade_date:
            exec_date = (pd.to_datetime(decision.trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        limit_price = getattr(decision, "planned_entry_price", 0.0) or 0.0

        # The only agent-generated order is a full-size BUY entry.
        target_qty = self._allocation_to_qty(
            decision, current_position, current_equity, current_cash, reference_price, limit_price
        )
        target_qty = self._round_to_lot(target_qty)
        target_qty = self._cap_by_leverage(target_qty, current_position, current_equity, reference_price)

        if target_qty <= 0:
            return []

        return self._build_orders(
            order_type=order_type,
            ticker=ticker,
            target_qty=target_qty,
            execution_date=exec_date,
            decision_id=decision.decision_id,
            current_position=current_position,
            limit_price=limit_price,
        )

    # ==================================================================
    # Legacy entry point (backward compat with old ParsedDecision)
    # ==================================================================
    def generate(
        self,
        decision: ParsedDecision,
        portfolio: any,
        execution_date: str,
        reference_point: MarketPoint,
        spec: InstrumentSpec,
    ) -> list[Order]:
        """Legacy entry point — wraps old ParsedDecision into ExtendedDecision.

        Supports both the old schema (decision_schema.ParsedDecision with .action,
        .rating) and the new schema (position.ParsedDecision / ExtendedDecision
        with .agent_rating). The new schema is detected by absence of .action.
        """
        if not decision.valid:
            return []

        # Resolve position attributes (support both Portfolio and PortfolioV2)
        underlying_pos = portfolio.position
        avg_price = getattr(
            underlying_pos, "avg_entry_price", getattr(underlying_pos, "avg_price", 0.0)
        )

        # Build a Position from portfolio
        pos = Position(
            ticker=portfolio.ticker,
            quantity=underlying_pos.quantity,
            avg_entry_price=avg_price,
            multiplier=underlying_pos.multiplier,
        )

        # Detect schema: old has .action; new has .agent_rating / .futures_action
        is_old_schema = hasattr(decision, "action") and hasattr(decision, "rating")

        if is_old_schema:
            return self._generate_from_old_schema(
                decision=decision,
                portfolio=portfolio,
                execution_date=execution_date,
                reference_point=reference_point,
                pos=pos,
            )

        # New schema (position.ParsedDecision / ExtendedDecision)
        return self._generate_from_new_schema(
            decision=decision,
            portfolio=portfolio,
            execution_date=execution_date,
            reference_point=reference_point,
            pos=pos,
        )

    def _generate_from_old_schema(
        self,
        decision: ParsedDecision,
        portfolio: any,
        execution_date: str,
        reference_point: MarketPoint,
        pos: Position,
    ) -> list[Order]:
        """Handle the legacy decision_schema.ParsedDecision (with .action, .rating)."""
        from .decision_schema import Action
        from .decision_schema import Rating as OldRating

        if (
            decision.action != Action.BUY
            or not decision.allow_new_position
            or not portfolio.is_flat()
        ):
            return []

        rating_map = {
            OldRating.BUY: "BUY",
            OldRating.OVERWEIGHT: "BUY",
            OldRating.HOLD: "WNS",
            OldRating.UNDERWEIGHT: "WNS",
            OldRating.SELL: "WNS",
            OldRating.WNS: "WNS",
        }
        futures_action = "BUY_TO_OPEN"

        alloc_pct = decision.target_position_pct
        if alloc_pct is not None and alloc_pct > 1.0:
            alloc_pct = alloc_pct / 100.0

        ext = ExtendedDecision(
            decision_id=decision.decision_id,
            ticker=decision.ticker,
            trade_date=decision.trade_date,
            report_generated_at=decision.report_generated_at,
            last_data_date=decision.last_data_date,
            decision_valid_from=decision.decision_valid_from,
            agent_rating=decision.rating.value,
            normalized_rating=rating_map.get(decision.rating, "WNS"),
            allocation_pct=alloc_pct,
            leverage=1.0,
            short_allowed=False,
            market_mode="SPOT_LONG_ONLY",
            allowed_position_sides="LONG",
            position_intent=decision.action.value.lower(),
            current_position_side=pos.side.value,
            target_position_side="LONG" if "BUY" in futures_action else pos.side.value,
            futures_action=futures_action,
            stop_price=decision.stop_price,
            take_profit=decision.take_profit,
            valid=decision.valid,
        )

        equity = portfolio.account_equity(reference_point.close)
        orders = self.decide(ext, pos, equity, reference_price=reference_point.close)

        for o in orders:
            o.execution_date = execution_date
        return orders

    def _generate_from_new_schema(
        self,
        decision: ParsedDecision,
        portfolio: any,
        execution_date: str,
        reference_point: MarketPoint,
        pos: Position,
    ) -> list[Order]:
        """Handle the new position.ParsedDecision / ExtendedDecision (with .agent_rating).

        The parser does not set position_intent or futures_action — those are
        produced by DecisionStateManager. We run the decision through the DSM
        first to map agent_rating → position_intent → futures_action, then
        delegate to decide().
        """
        from .decision_state_manager import DecisionStateManager

        dsm = DecisionStateManager(self._dsm_config)
        ext = dsm.map(decision=decision, current_position=pos)

        equity = portfolio.account_equity(reference_point.close)
        orders = self.decide(
            ext, pos, equity, reference_price=reference_point.close
        )

        for o in orders:
            o.execution_date = execution_date
        return orders

    # ==================================================================
    # Order building
    # ==================================================================
    def _build_orders(
        self,
        order_type: OrderType,
        ticker: str,
        target_qty: int,
        execution_date: str,
        decision_id: str,
        current_position: Position,
        limit_price: float = 0.0,
    ) -> list[Order]:
        reason_map = {
            OrderType.BUY_TO_OPEN: "agent_buy_open",
            OrderType.SELL_TO_CLOSE: "agent_sell_close",
        }

        reason = "agent_buy_limit" if limit_price > 0 else reason_map.get(order_type, "agent_order")

        return [
            Order(
                order_id=str(uuid4()),
                decision_id=decision_id,
                ticker=ticker,
                order_type=order_type,
                quantity=target_qty,
                price=limit_price,
                execution_date=execution_date,
                reason=reason,
            )
        ]

    # ==================================================================
    # Sizing helpers
    # ==================================================================
    def _allocation_to_qty(
        self,
        decision: ExtendedDecision,
        position: Position,
        equity: float,
        current_cash: Optional[float] = None,
        reference_price: float = 0.0,
        limit_price: float = 0.0,
    ) -> int:
        """Convert allocation_pct to a share quantity.

        For OPEN: initial_entry_pct is applied to equity. Existing long
        positions never create another entry.
        """
        alloc = decision.allocation_pct
        if alloc is None or alloc <= 0:
            dm = getattr(self.config, "decision_mapping", None)
            alloc = dm.initial_entry_pct if dm is not None else 0.25

        # Guard against percentage values (e.g., 30 instead of 0.30)
        if alloc > 1.0:
            alloc = alloc / 100.0

        # For OPEN: apply percentage to equity clamped by available cash in spot mode
        target_notional = equity * alloc

        # Spot Cash Clamp: In spot long-only mode, cannot spend more than free liquid cash
        if getattr(self.margin_cfg, "spot_mode", True) or getattr(self.margin_cfg, "initial_margin_pct", 1.0) >= 1.0:
            free_cash = max(0.0, current_cash if current_cash is not None else float(getattr(position, "cash", equity)))
            if hasattr(self, "_current_cash") and self._current_cash is not None and current_cash is None:
                free_cash = max(0.0, self._current_cash)
            buy_fee = getattr(self.exec_cfg, "buy_fee", getattr(self.exec_cfg, "buy_fee_pct", 0.001))
            slippage = getattr(self.exec_cfg, "slippage", getattr(self.exec_cfg, "slippage_pct", 0.0005))
            fee_factor = 1.0 + float(buy_fee) + float(slippage)
            target_notional = min(target_notional, free_cash / fee_factor)

        price_ref = limit_price if limit_price > 0 else (
            reference_price
            if reference_price > 0
            else (
                position.mark_price
                if position.mark_price and position.mark_price > 0
                else (decision.stop_price if decision.stop_price and decision.stop_price > 0 else 100.0)
            )
        )
        if not price_ref or price_ref <= 0:
            logger.warning("Cannot size order: no reference price available for %s", decision.ticker)
            return 0
        multiplier = getattr(self.exec_cfg, "contract_multiplier", 1.0)
        qty = int(target_notional // (price_ref * multiplier))
        return max(0, qty)

    def _round_to_lot(self, qty: int) -> int:
        """Round quantity down to nearest lot_size."""
        lot = getattr(self.exec_cfg, "lot_size", 1)
        if lot <= 1:
            return qty
        return (qty // lot) * lot

    def _cap_by_leverage(
        self,
        qty: int,
        position: Position,
        equity: float,
        reference_price: float = 0.0,
    ) -> int:
        """Clamp qty so total exposure <= equity * max_leverage."""
        max_lev = getattr(self.margin_cfg, "max_leverage", 2.0)
        if max_lev <= 0:
            return qty

        price_ref = reference_price if reference_price > 0 else (position.mark_price or 100.0)
        existing_exposure = position.notional(price_ref)
        max_new_exposure = max(0.0, equity * max_lev - existing_exposure)
        if max_new_exposure <= 0:
            return 0

        multiplier = getattr(self.exec_cfg, "contract_multiplier", 1.0)
        max_qty_by_lev = int(max_new_exposure // (price_ref * multiplier))
        return min(qty, max(0, max_qty_by_lev))

    # ==================================================================
    # Stop / liquidation order generators
    # ==================================================================
    def generate_stop_order(
        self,
        decision_id: str,
        portfolio: any,
        execution_date: str,
        reason: str = "stop_loss",
    ) -> list[Order]:
        if not portfolio.has_position() or not portfolio.is_long():
            return []
        order_type = OrderType.SELL_TO_CLOSE
        return [
            Order(
                order_id=str(uuid4()),
                decision_id=decision_id,
                ticker=portfolio.ticker,
                order_type=order_type,
                quantity=portfolio.abs_qty(),
                execution_date=execution_date,
                reason=reason,
                is_risk_order=True,
            )
        ]

    def generate_liquidation_order(
        self,
        portfolio: any,
        execution_date: str,
    ) -> Optional[Order]:
        if not portfolio.has_position() or not portfolio.is_long():
            return None
        order_type = OrderType.SELL_TO_CLOSE
        return Order(
            order_id=str(uuid4()),
            decision_id="liquidation",
            ticker=portfolio.ticker,
            order_type=order_type,
            quantity=portfolio.abs_qty(),
            execution_date=execution_date,
            reason="liquidation_margin_breach",
            is_risk_order=True,
        )
