"""
Order generation for stock margin backtests — PRD-compliant version.

Translates an ExtendedDecision (or legacy ParsedDecision) into one or more
pending Orders with explicit OrderType (PRD §7.2):
- BUY_TO_OPEN, BUY_TO_ADD, SELL_TO_REDUCE, SELL_TO_CLOSE
- SELL_TO_OPEN, SELL_TO_ADD, BUY_TO_REDUCE, BUY_TO_CLOSE
- REVERSE_TO_LONG, REVERSE_TO_SHORT, NO_ORDER

Supports:
- Lot-size rounding
- Max-leverage cap enforcement
- Two-leg reverse execution (close leg + open leg)
- Percentage-based fees/slippage
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

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

        ticker = decision.ticker
        exec_date = decision.decision_valid_from or decision.trade_date

        # Calculate target quantity from allocation_pct
        target_qty = self._allocation_to_qty(decision, current_position, current_equity, reference_price)

        # Lot-size rounding
        target_qty = self._round_to_lot(target_qty)

        # Max-leverage cap
        target_qty = self._cap_by_leverage(target_qty, current_position, current_equity)

        # For close orders, use full position quantity if allocation yielded 0
        if target_qty <= 0 and order_type in (
            OrderType.SELL_TO_CLOSE, OrderType.BUY_TO_CLOSE,
        ):
            target_qty = current_position.abs_qty()

        if target_qty <= 0:
            return []

        return self._build_orders(
            order_type=order_type,
            ticker=ticker,
            target_qty=target_qty,
            execution_date=exec_date,
            decision_id=decision.decision_id,
            current_position=current_position,
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
        from .decision_schema import Action, Rating as OldRating

        if decision.action == Action.INVALID:
            return []
        if decision.action == Action.HOLD:
            return []

        rating_map = {
            OldRating.BUY: "strong_buy",
            OldRating.OVERWEIGHT: "buy",
            OldRating.HOLD: "hold",
            OldRating.UNDERWEIGHT: "sell",
            OldRating.SELL: "strong_sell",
        }
        action_map = {
            Action.BUY: "BUY_TO_OPEN",
            Action.ADD: "BUY_TO_ADD",
            Action.SELL: "SELL_TO_CLOSE",
            Action.OPEN_SHORT: "SELL_TO_OPEN",
            Action.COVER_SHORT: "BUY_TO_CLOSE",
            Action.REDUCE: "SELL_TO_REDUCE" if portfolio.is_long() else "BUY_TO_REDUCE",
            Action.SELL_ALL: "SELL_TO_CLOSE" if portfolio.is_long() else "BUY_TO_CLOSE",
        }

        from .decision_schema import Action as LegacyAction
        futures_action = action_map.get(decision.action, "NO_ORDER")

        # Legacy: BUY closes short position, SELL/OPEN_SHORT closes long
        if portfolio.is_short() and decision.action == LegacyAction.BUY:
            futures_action = "BUY_TO_CLOSE"
        elif portfolio.is_long() and decision.action in (LegacyAction.SELL, LegacyAction.OPEN_SHORT):
            futures_action = "SELL_TO_CLOSE"

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
            normalized_rating=rating_map.get(decision.rating, "hold"),
            allocation_pct=alloc_pct,
            leverage=1.0,
            short_allowed=True,
            market_mode="FUTURES_STYLE_SIMULATION",
            allowed_position_sides="LONG,SHORT",
            position_intent=decision.action.value.lower(),
            current_position_side=pos.side.value,
            target_position_side="LONG" if "BUY" in futures_action else ("SHORT" if "SELL_TO_OPEN" in futures_action else pos.side.value),
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
    ) -> list[Order]:
        if order_type in (OrderType.REVERSE_TO_LONG, OrderType.REVERSE_TO_SHORT):
            # Two-leg reverse: close existing, then open new
            close_qty = current_position.abs_qty()
            close_type = (
                OrderType.SELL_TO_CLOSE if current_position.is_long()
                else OrderType.BUY_TO_CLOSE
            )

            close_order = Order(
                order_id=str(uuid4()),
                decision_id=decision_id,
                ticker=ticker,
                order_type=close_type,
                quantity=close_qty,
                execution_date=execution_date,
                reason="reverse_close_leg",
            )

            open_type = (
                OrderType.BUY_TO_OPEN if order_type == OrderType.REVERSE_TO_LONG
                else OrderType.SELL_TO_OPEN
            )
            open_order = Order(
                order_id=str(uuid4()),
                decision_id=decision_id,
                ticker=ticker,
                order_type=open_type,
                quantity=target_qty,
                execution_date=execution_date,
                reason="reverse_open_leg",
                is_reverse=True,
            )
            return [close_order, open_order]

        reason_map = {
            OrderType.BUY_TO_OPEN: "agent_buy_open",
            OrderType.BUY_TO_ADD: "agent_buy_add",
            OrderType.SELL_TO_REDUCE: "agent_sell_reduce",
            OrderType.SELL_TO_CLOSE: "agent_sell_close",
            OrderType.SELL_TO_OPEN: "agent_open_short",
            OrderType.SELL_TO_ADD: "agent_short_add",
            OrderType.BUY_TO_REDUCE: "agent_cover_reduce",
            OrderType.BUY_TO_CLOSE: "agent_cover_close",
        }

        return [
            Order(
                order_id=str(uuid4()),
                decision_id=decision_id,
                ticker=ticker,
                order_type=order_type,
                quantity=target_qty,
                execution_date=execution_date,
                reason=reason_map.get(order_type, "agent_order"),
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
        reference_price: float = 0.0,
    ) -> int:
        """Convert allocation_pct to a share quantity.

        For REDUCE: reduce_step_pct is applied to current position quantity.
        For INCREASE: pyramid_pct is applied to current position quantity.
        For OPEN: initial_entry_pct is applied to equity.
        """
        alloc = decision.allocation_pct
        if alloc is None or alloc <= 0:
            dm = getattr(self.config, "decision_mapping", None)
            if dm is not None:
                if decision.position_intent == "increase":
                    alloc = dm.pyramid_pct
                elif decision.position_intent == "reduce":
                    alloc = dm.reduce_step_pct
                else:
                    alloc = dm.initial_entry_pct
            else:
                alloc = 0.25

        # Guard against percentage values (e.g., 30 instead of 0.30)
        if alloc > 1.0:
            alloc = alloc / 100.0

        # For REDUCE: apply percentage to current position quantity
        if decision.position_intent == "reduce" and position.abs_qty() > 0:
            return max(1, int(position.abs_qty() * alloc))

        # For INCREASE: apply percentage to current position quantity
        if decision.position_intent == "increase" and position.abs_qty() > 0:
            return max(1, int(position.abs_qty() * alloc))

        # For OPEN: apply percentage to equity
        target_notional = equity * alloc
        price_ref = reference_price or decision.stop_price
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
    ) -> int:
        """Clamp qty so total exposure <= equity * max_leverage."""
        max_lev = getattr(self.margin_cfg, "max_leverage", 2.0)
        if max_lev <= 0:
            return qty

        existing_exposure = position.notional(position.mark_price or 0.0)
        max_new_exposure = equity * max_lev - existing_exposure
        if max_new_exposure <= 0:
            return 0

        price_ref = position.mark_price or 100.0
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
        if not portfolio.has_position():
            return []
        order_type = (
            OrderType.SELL_TO_CLOSE if portfolio.is_long()
            else OrderType.BUY_TO_CLOSE
        )
        return [
            Order(
                order_id=str(uuid4()),
                decision_id=decision_id,
                ticker=portfolio.ticker,
                order_type=order_type,
                quantity=portfolio.abs_qty(),
                execution_date=execution_date,
                reason=reason,
            )
        ]

    def generate_liquidation_order(
        self,
        portfolio: any,
        execution_date: str,
    ) -> Optional[Order]:
        if not portfolio.has_position():
            return None
        order_type = (
            OrderType.SELL_TO_CLOSE if portfolio.is_long()
            else OrderType.BUY_TO_CLOSE
        )
        return Order(
            order_id=str(uuid4()),
            decision_id="liquidation",
            ticker=portfolio.ticker,
            order_type=order_type,
            quantity=portfolio.abs_qty(),
            execution_date=execution_date,
            reason="liquidation_margin_breach",
        )
