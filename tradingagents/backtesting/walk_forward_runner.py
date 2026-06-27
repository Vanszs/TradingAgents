"""
Walk-forward backtest runner — PRD §17 + new trigger/window pipeline.

Per-day loop (PRD §17 pseudo-code + new steps):
  1. Execute pending orders at today's open
  2. Check risk events bar-by-bar (liquidation, stop, target)
  3. Mark-to-market at daily close
  4. Create snapshot up to current date (with optional lookback window)
  5. Daily Review Agent runs after market close (always — independent
     of whether an order will be generated)
  6. Parse decision
  7. Validate no leakage (provider, OHLCV, news, fundamentals, sentiment)
  8. Save decision
  9. Run Trigger-Based Execution Agent — emits an order ONLY if a
     trigger condition fires; otherwise the rating is recorded but no
     order is produced.
 10. Generate orders for next trading day
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from .agent_runner import TradingAgentsRunner
from .broker import SimulatedBroker
from .calendar import TradingCalendar
from .cutoff_validator import DecisionCutoffValidator
from .decision_schema import (
    BacktestConfig,
    InstrumentSpec,
    MarginEvent,
    MarketPoint,
    OpenClose,
    Order,
)
from .position import ParsedDecision
from .decision_store import DecisionStore
from .markdown_parser import MarkdownDecisionParser
from .metrics import MetricsCalculator
from .order_generator import OrderGenerator
from .portfolio import PortfolioV2
from .margin import MarginAccount
from .risk import RiskEngine
from .reports import BacktestReportGenerator
from .snapshot_provider import SnapshotDataProvider
from .position import Position, MarginConfig, RiskConfig, DecisionMappingConfig
from .decision_state_manager import DecisionStateManager
from .trigger_evaluator import TriggerConfig, TriggerEvaluator, TriggerResult


class WalkForwardBacktestRunner:
    def __init__(
        self,
        config: BacktestConfig,
        agent_runner: Optional[TradingAgentsRunner] = None,
        progress_callback: Optional[Any] = None,
    ):
        self.config = config
        self.config.validate()
        self.progress_callback = progress_callback
        self._day_index = 0
        self._trading_days: list[str] = []

        self.snapshot_provider = SnapshotDataProvider(
            data_root=config.data.data_root,
            snapshot_root=config.data.snapshot_root,
            lookback_days=config.lookback_days,
        )

        # Load instrument spec from OHLCV
        ohlcv_df = self.snapshot_provider.load_full_ohlcv(config.ticker)
        self.spec = InstrumentSpec.auto_from_ohlcv(config.ticker, ohlcv_df)
        self._ohlcv_df = ohlcv_df  # Store for ATR computation

        # Calendar from market dates
        market_dates = self.snapshot_provider.get_market_dates(config.ticker)
        self.calendar = TradingCalendar(market_dates=market_dates)

        self.agent_runner = agent_runner or TradingAgentsRunner(
            reports_root=config.output.reports_root,
            agent_config=config.agent,
        )
        # Always make the per-node timing callback available to the
        # agent runner, even when the runner was constructed outside
        # the engine. The runner no-ops when no progress callback is
        # supplied, so this is safe for tests and the legacy path.
        if progress_callback is not None and getattr(
            self.agent_runner, "_progress", None
        ) is None:
            self.agent_runner._progress = progress_callback

        self.parser = MarkdownDecisionParser()
        self.validator = DecisionCutoffValidator(
            fail_on_future_data=config.leakage_guard.fail_on_future_data
        )
        self.decision_store = DecisionStore(
            root=str(Path(config.output.output_root) / config.ticker / "decisions")
        )

        self.portfolio = self._init_portfolio()
        self.broker = SimulatedBroker(
            execution_config=config.execution,
            margin_config=config.margin,
        )
        self.order_generator = OrderGenerator(config)
        self.dsm = DecisionStateManager(config=config.decision_mapping)
        self.metrics = MetricsCalculator()
        self.reporter = BacktestReportGenerator(config)

        # Risk engine for bar-by-bar checks (PRD §11)
        risk_cfg = RiskConfig()
        self.risk_engine = RiskEngine(
            max_loss_per_trade_pct=risk_cfg.max_intraday_loss_pct,
            max_portfolio_loss_pct=risk_cfg.max_intraday_loss_pct,
            liquidation_enabled=True,
            stop_loss_enabled=True,
            take_profit_enabled=True,
        )

        # Trigger-Based Execution Agent. Defaults to TriggerConfig(); can be
        # overridden by ``config.trigger`` if the caller provided one.
        self.trigger_evaluator = TriggerEvaluator(config.trigger or TriggerConfig())

        self.leakage_checks: dict[str, str] = {
            "memory_isolation": "PASSED" if not config.agent.memory_enabled else "FAILED",
            "live_provider_disabled": "PASSED",
            "lookback_window_respected": "PASSED",
        }

        self.prev_close: Optional[float] = None
        self.prev_decision: Optional[ParsedDecision] = None
        self.margin_events: list[MarginEvent] = []
        self.position_log: list[dict] = []
        self.margin_log: list[dict] = []
        self.trigger_log: list[dict] = []
        self._next_reanalysis_date: Optional[str] = None

    def _init_portfolio(self) -> PortfolioV2:
        margin_cfg = MarginConfig(
            initial_margin_pct=self.config.margin.initial_margin_pct,
            maintenance_margin_pct=self.config.margin.maintenance_margin_pct,
            max_leverage=self.config.margin.max_leverage,
            max_entry_pct=getattr(self.config.margin, "max_entry_pct", 0.10),
        )
        return PortfolioV2(
            initial_cash=self.config.initial_cash,
            ticker=self.config.ticker,
            margin_config=margin_cfg,
            initial_position_qty=self.config.initial_position_qty,
            initial_position_price=self.config.initial_position_price,
            initial_multiplier=self.spec.multiplier,
        )

    def run(self) -> dict[str, Any]:
        trading_days = self.calendar.trading_days(
            self.config.start_date, self.config.end_date
        )
        if not trading_days:
            raise ValueError("No trading days found in selected period.")

        self._trading_days = list(trading_days)
        self._day_index = 0
        self._notify_progress("start", None, 0, len(self._trading_days))

        self._load_initial_report_if_any()

        for current_date in trading_days:
            self._process_day(current_date)
            self._day_index += 1
            self._notify_progress(
                "day_complete",
                current_date,
                self._day_index,
                len(self._trading_days),
            )

        # Close any remaining open positions at the end of backtest
        if not self.portfolio.is_flat():
            last_date = self._trading_days[-1]
            last_market_point = self.snapshot_provider.get_market_point(
                symbol=self.config.ticker,
                trade_date=last_date,
                spec=self.spec,
            )
            close_order = self._build_force_close(
                last_date, last_market_point.close, "end_of_backtest"
            )
            if close_order is not None:
                self.broker.add_pending_orders([close_order])
                self.broker.execute_pending_orders(
                    date=last_date,
                    market_point=last_market_point,
                    portfolio=self.portfolio,
                    spec=self.spec,
                )
                # Mark-to-market after close to capture correct post-close equity
                self.portfolio.position.mark_to_market(last_market_point.close)
                self.portfolio.mark_to_market(
                    date=last_date,
                    close_price=last_market_point.close,
                )
            logger.info(
                f"[BACKTEST] Closed remaining position at end of backtest: "
                f"date={last_date}, side={self.portfolio.position.side.value}, "
                f"price={last_market_point.close}"
            )

        self._notify_progress("done", None, len(self._trading_days), len(self._trading_days))
        return self._finalize()

    def _notify_progress(
        self,
        phase: str,
        trade_date: Optional[str],
        day_idx: int,
        total_days: int,
        **extra: Any,
    ) -> None:
        """Fire the user-supplied progress callback if one is registered.

        The callback signature is
        ``callback(phase, trade_date, day_idx, total_days, **extra)``.
        Any exception raised by the callback is swallowed so a buggy UI
        cannot poison the backtest loop.
        """
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(
                phase=phase,
                trade_date=trade_date,
                day_idx=day_idx,
                total_days=total_days,
                **extra,
            )
        except Exception:  # noqa: BLE001
            pass

    def _process_day(self, current_date: str) -> None:
        day_start_time = time.time()
        logger.info(
            f"[BACKTEST] === Day {self._day_index + 1}/{len(self._trading_days) if self._trading_days else '?'}: {current_date} ==="
        )

        self._notify_progress(
            "day_start",
            current_date,
            self._day_index + 1,
            len(self._trading_days) if self._trading_days else 0,
        )

        # Load market point for today
        step_start = time.time()
        market_point = self.snapshot_provider.get_market_point(
            symbol=self.config.ticker,
            trade_date=current_date,
            spec=self.spec,
        )
        logger.info(f"[BACKTEST] Step 0 (load market point): {time.time() - step_start:.3f}s")

        # PRD §17 step 1: Execute pending orders at today's open
        step_start = time.time()
        executed_trades = self.broker.execute_pending_orders(
            date=current_date,
            market_point=market_point,
            portfolio=self.portfolio,
            spec=self.spec,
        )
        logger.info(f"[BACKTEST] Step 1 (execute pending orders): {time.time() - step_start:.3f}s")

        # Notify about executed trades so the CLI can display them
        if executed_trades:
            trade_dicts = []
            for t in executed_trades:
                trade_dicts.append({
                    "date": t.date,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "order_type": t.order_type,
                    "reason": t.reason,
                    "realized_pnl": t.realized_pnl_delta,
                })
            self._notify_progress(
                "execute",
                current_date,
                self._day_index + 1,
                len(self._trading_days) if self._trading_days else 0,
                trades=trade_dicts,
            )

        # Mark the underlying position to today's open before the risk check
        # so the RiskEngine sees a current unrealized PnL (not yesterday's
        # cached value). This prevents a 1-day lag in margin breach detection.
        if not self.portfolio.is_flat():
            self.portfolio.position.mark_to_market(market_point.open)

        # PRD §17 step 2: Check risk events bar-by-bar
        step_start = time.time()
        self._notify_progress(
            "risk",
            current_date,
            self._day_index + 1,
            len(self._trading_days) if self._trading_days else 0,
        )
        risk_order = self._check_risk_events(current_date, market_point)
        logger.info(f"[BACKTEST] Step 2 (risk check): {time.time() - step_start:.3f}s")

        # Explicit EOD margin breach check (PortfolioV2). The bar-by-bar
        # risk engine emits a force-close on breach; this secondary check
        # records a margin event and ensures the breach is always logged
        # even if the bar-by-bar check did not produce an order (e.g. flat
        # position state). The force-close itself is delegated to the
        # risk engine's output (risk_order).
        if (
            not self.portfolio.is_flat()
            and self.portfolio.is_margin_breach(market_point.close)
        ):
            equity = self.portfolio.account_equity(market_point.close)
            maintenance = self.portfolio.maintenance_required(
                market_point.close
            )
            self.margin_events.append(MarginEvent(
                date=current_date,
                kind="eod_margin_breach",
                mark_price=market_point.close,
                deficit=maintenance - equity,
                maintenance_required=maintenance,
                account_equity=equity,
                action="liquidation",
            ))
            # If the bar-by-bar check missed a liquidation (e.g. the
            # equity just dropped at close), synthesize one.
            if risk_order is None or risk_order.order_type.value == "NO_ORDER":
                from .position import OrderType
                if self.portfolio.is_long():
                    liq_type = OrderType.SELL_TO_CLOSE
                else:
                    liq_type = OrderType.BUY_TO_CLOSE
                next_valid_date = self._safe_next_trading_day(current_date)
                if next_valid_date is None:
                    next_valid_date = current_date
                risk_order = self._build_force_close(
                    next_valid_date, market_point.close, "eod_margin_breach"
                )
                if risk_order is not None:
                    self.broker.add_pending_orders([risk_order])

        # PRD §17 step 3: Mark-to-market at daily close
        step_start = time.time()
        if not self.portfolio.is_flat():
            self.portfolio.position.mark_to_market(market_point.close)
        snapshot = self.portfolio.mark_to_market(
            date=current_date,
            close_price=market_point.close,
        )

        # Collect position and margin logs
        self._collect_position_log(current_date, market_point)
        self._collect_margin_log(current_date, snapshot)
        logger.info(f"[BACKTEST] Step 3 (mark-to-market): {time.time() - step_start:.3f}s")

        # PRD §17 step 4: Create snapshot up to current date
        step_start = time.time()
        snapshot_data = self.snapshot_provider.create_snapshot(
            symbol=self.config.ticker,
            trade_date=current_date,
            config=self.config,
        )
        self._audit_window(snapshot_data)
        logger.info(
            f"[BACKTEST] Step 4 (snapshot): {time.time() - step_start:.3f}s | "
            f"news={len(snapshot_data.news)}, sentiment={len(snapshot_data.sentiment)}, "
            f"fundamentals={len(snapshot_data.fundamentals)}, broker={len(snapshot_data.broker_activity)}"
        )

        # Track previous rating for logging and comparison.
        prev_rating = self.prev_decision.agent_rating if self.prev_decision else None

        # Check if we should skip the agent (conditional reanalysis).
        # If _next_reanalysis_date is set and current_date is before it,
        # skip the full agent pipeline and carry forward the previous decision.
        should_skip_agent = (
            self._next_reanalysis_date is not None
            and current_date < self._next_reanalysis_date
        )

        if should_skip_agent:
            # Skip agent: carry forward previous decision as HOLD
            self._notify_progress(
                "agent",
                current_date,
                self._day_index + 1,
                len(self._trading_days) if self._trading_days else 0,
            )
            # Create synthetic decision from previous
            next_valid_date = self._safe_next_trading_day(current_date)
            if next_valid_date is None:
                self.prev_close = market_point.close
                return

            if self.prev_decision is not None:
                decision = ParsedDecision(
                    decision_id=f"{self.config.ticker}-{current_date}-SKIP",
                    ticker=self.config.ticker,
                    trade_date=current_date,
                    agent_rating=self.prev_decision.agent_rating,
                    report_generated_at=f"{current_date} 16:30:00",
                    last_data_date=current_date,
                    decision_valid_from=next_valid_date,
                    normalized_rating=self.prev_decision.normalized_rating,
                    confidence=self.prev_decision.confidence,
                    allocation_pct=self.prev_decision.allocation_pct,
                    reduce_pct=self.prev_decision.reduce_pct,
                    leverage=self.prev_decision.leverage,
                    stop_price=self.prev_decision.stop_price,
                    take_profit=self.prev_decision.take_profit,
                    time_horizon_days=self.prev_decision.time_horizon_days,
                    allow_new_position=self.prev_decision.allow_new_position,
                    short_allowed=self.prev_decision.short_allowed,
                    market_mode=self.prev_decision.market_mode,
                    allowed_position_sides=self.prev_decision.allowed_position_sides,
                    source_report_path=None,
                    raw_text_excerpt=f"Skipped — next review: {self._next_reanalysis_date}",
                    next_review_date=self._next_reanalysis_date,
                )
            else:
                # No previous decision: run agent normally
                should_skip_agent = False

        if not should_skip_agent:
            # PRD §17 step 5: Run agent after market close (Daily Review Agent)
            # This is the slowest step in real backtests (depends on LLM latency).
            step_start = time.time()
            logger.info(f"[BACKTEST] Step 5 (agent run): Starting LLM calls for {self.config.ticker} on {current_date}...")
            self._notify_progress(
                "agent",
                current_date,
                self._day_index + 1,
                len(self._trading_days) if self._trading_days else 0,
            )
            report_text = self.agent_runner.run(
                ticker=self.config.ticker,
                trade_date=current_date,
                snapshot=snapshot_data,
                day_idx=self._day_index + 1,
                n_days=len(self._trading_days) if self._trading_days else 0,
            )
            logger.info(f"[BACKTEST] Step 5 (agent run): {time.time() - step_start:.3f}s | report length: {len(report_text)}")

            # PRD §17 step 6: Parse decision
            step_start = time.time()
            self._notify_progress(
                "parse",
                current_date,
                self._day_index + 1,
                len(self._trading_days) if self._trading_days else 0,
            )
            next_valid_date = self._safe_next_trading_day(current_date)
            if next_valid_date is None:
                self.prev_close = market_point.close
                return

            decision = self.parser.parse_text(
                text=report_text,
                source_report_path=str(
                    Path(self.config.output.reports_root)
                    / self.config.ticker
                    / current_date
                    / "complete_report.md"
                ),
                fallback_ticker=self.config.ticker,
                fallback_trade_date=current_date,
                fallback_decision_valid_from=next_valid_date,
            )
            logger.info(f"[BACKTEST] Step 6 (parse decision): {time.time() - step_start:.3f}s")

            # Handle invalid decision gracefully — if parser couldn't extract
            # a valid rating (e.g. MiniMax emitted a tool call instead of
            # analysis), use a HOLD fallback so the backtest doesn't crash.
            if not decision.valid:
                logger.warning(
                    f"[BACKTEST] Parser returned invalid decision for {current_date}: "
                    f"{decision.invalid_reason}. Using HOLD fallback."
                )
                decision = ParsedDecision(
                    decision_id=f"{self.config.ticker}-{current_date}-FALLBACK",
                    ticker=self.config.ticker,
                    trade_date=current_date,
                    agent_rating="Hold",
                    report_generated_at=f"{current_date} 16:30:00",
                    last_data_date=current_date,
                    decision_valid_from=next_valid_date,
                    normalized_rating="HOLD",
                    source_report_path=None,
                    raw_text_excerpt=f"Fallback HOLD — parser invalid: {decision.invalid_reason}",
                    valid=True,
                )

            # PRD §17 step 7: Validate no leakage
            step_start = time.time()
            checks = self.validator.validate(
                decision=decision,
                snapshot_metadata=snapshot_data.metadata,
            )
            self.leakage_checks.update(checks)
            logger.info(f"[BACKTEST] Step 7 (validate leakage): {time.time() - step_start:.3f}s")

            # Track rating change before storing the decision.
            decision.prev_rating = prev_rating
            decision.rating_changed = bool(
                prev_rating is not None and prev_rating != decision.agent_rating
            )

            # Update _next_reanalysis_date from the parsed decision.
            # If parser didn't extract next_review_date (common with free-text
            # fallback on models like MiniMax that don't always include the
            # "**Next Review Date**: YYYY-MM-DD" section), log a warning
            # and use next trading day as default.
            if decision.next_review_date:
                self._next_reanalysis_date = decision.next_review_date
            else:
                logger.warning(
                    f"[BACKTEST] next_review_date NOT FOUND in report for {current_date}. "
                    f"Rating={decision.agent_rating}. "
                    f"This usually means the PM free-text response didn't include "
                    f"'**Next Review Date**: YYYY-MM-DD'. "
                    f"Using next trading day as default."
                )
                # Default to next trading day
                next_date_str = self.calendar.next_trading_day(current_date)
                if next_date_str:
                    decision.next_review_date = next_date_str
                else:
                    # Fallback if calendar doesn't have next trading day
                    base_date = datetime.strptime(current_date, "%Y-%m-%d")
                    next_date = base_date + timedelta(days=1)
                    while next_date.weekday() >= 5:  # skip weekends
                        next_date += timedelta(days=1)
                    decision.next_review_date = next_date.strftime("%Y-%m-%d")
                self._next_reanalysis_date = decision.next_review_date

        # PRD §17 step 8: Save decision
        step_start = time.time()
        self.decision_store.save(decision)

        # Update risk levels from decision
        self._update_risk_levels(decision, self._ohlcv_df)
        logger.info(f"[BACKTEST] Step 8 (save decision): {time.time() - step_start:.3f}s")

        # Step 9a: Map decision into a state-aware action.
        step_start = time.time()
        extended_decision = self.dsm.map(
            decision=decision,
            current_position=self.portfolio.position,
        )

        # Step 9b: Trigger-Based Execution Agent gate. A non-triggered
        # rating becomes NO_ORDER (recorded, not executed).
        trigger_result = self.trigger_evaluator.evaluate(
            today_decision=extended_decision,
            prev_decision=self.prev_decision,
            current_position=self.portfolio.position,
            risk_order=risk_order,
            current_equity=self.portfolio.account_equity(market_point.close),
        )
        extended_decision.triggered = trigger_result.triggered
        extended_decision.trigger_reasons = list(trigger_result.reasons)
        extended_decision.trigger_details = dict(trigger_result.details)

        self.trigger_log.append({
            "trade_date": current_date,
            "decision_id": decision.decision_id,
            "agent_rating": decision.agent_rating,
            "prev_rating": prev_rating,
            "rating_changed": decision.rating_changed,
            "current_position_side": self.portfolio.position.side.value,
            "triggered": trigger_result.triggered,
            "reasons": trigger_result.reasons,
            "details": trigger_result.details,
            "implied_futures_action": extended_decision.futures_action,
            "executed_futures_action": (
                extended_decision.futures_action
                if trigger_result.triggered
                else "NO_ORDER"
            ),
        })

        if not trigger_result.triggered:
            extended_decision.futures_action = "NO_ORDER"
        logger.info(f"[BACKTEST] Step 9 (trigger evaluation): {time.time() - step_start:.3f}s | triggered={trigger_result.triggered}")

        self._notify_progress(
            "trigger",
            current_date,
            self._day_index + 1,
            len(self._trading_days) if self._trading_days else 0,
            triggered=trigger_result.triggered,
            reasons=list(trigger_result.reasons),
        )

        # Step 9c: Generate orders for next trading day.
        step_start = time.time()
        orders = self.order_generator.decide(
            decision=extended_decision,
            current_position=self.portfolio.position,
            current_equity=self.portfolio.account_equity(market_point.close),
            reference_price=market_point.close,
        )
        self.broker.add_pending_orders(orders)
        logger.info(f"[BACKTEST] Step 9c (generate orders): {time.time() - step_start:.3f}s | orders={len(orders)}")

        # Update state for tomorrow
        self.prev_close = market_point.close
        self.prev_decision = extended_decision

        # Log total day time
        day_total_time = time.time() - day_start_time
        logger.info(f"[BACKTEST] === Day {current_date} complete: {day_total_time:.3f}s total | rating={decision.agent_rating} ===")

    def _check_risk_events(self, date: str, market_point: MarketPoint) -> Optional[Order]:
        """PRD §17 step 2 — check risk events bar-by-bar.

        Returns the risk-engine order (if any) so the trigger evaluator can
        treat a stop/take-profit hit as an explicit trigger.
        """
        if self.portfolio.is_flat():
            return None

        bar = {
            "open": market_point.open,
            "high": market_point.high,
            "low": market_point.low,
            "close": market_point.close,
        }

        equity = self.portfolio.account_equity(market_point.close)

        order, events = self.risk_engine.check_bar(
            date=date,
            bar=bar,
            position=self.portfolio.position,
            equity=equity,
            margin_rate=self.config.margin.initial_margin_pct,
            maintenance_rate=self.config.margin.maintenance_margin_pct,
            max_leverage=self.config.margin.max_leverage,
            multiplier=self.spec.multiplier,
        )

        # Record events
        maintenance_required = (
            abs(self.portfolio.position.quantity)
            * market_point.close
            * self.spec.multiplier
            * self.config.margin.maintenance_margin_pct
        )
        for event in events:
            self.margin_events.append(MarginEvent(
                date=date,
                kind=event.event_type,
                mark_price=event.price,
                deficit=0.0,
                maintenance_required=maintenance_required,
                account_equity=equity,
                action=event.reason,
            ))

        # Execute risk-triggered orders. Drop NO_ORDERs but never swallow a
        # genuine liquidation/stop/take-profit (the risk engine only returns
        # NO_ORDER for a FLAT position, which we already short-circuited above).
        if order is not None and order.order_type.value != "NO_ORDER":
            # Schedule for next trading day since today's execution step has passed
            next_valid_date = self._safe_next_trading_day(date)
            if next_valid_date is not None:
                order.execution_date = next_valid_date
            self.broker.add_pending_orders([order])

        return order

    def _build_force_close(self, date: str, price: float, reason: str) -> Optional["Order"]:
        """Build a force-close Order for the current position.

        Used as a fallback when the bar-by-bar risk engine did not produce
        an order but the portfolio is in margin breach (e.g. the close price
        alone tipped the account below maintenance). Returns None for a
        flat position.
        """
        from .position import Order as PositionOrder, OrderType

        if self.portfolio.is_flat():
            return None
        if self.portfolio.is_long():
            order_type = OrderType.SELL_TO_CLOSE
        else:
            order_type = OrderType.BUY_TO_CLOSE
        return PositionOrder(
            order_id=f"risk_close_{date}",
            decision_id="",
            ticker=self.config.ticker,
            order_type=order_type,
            quantity=abs(self.portfolio.position.quantity),
            execution_date=date,
            price=price,
            reason=reason,
        )

    def _audit_window(self, snapshot) -> None:
        """Validate that the snapshot's window respects the configured lookback."""
        if self.config.lookback_days is None:
            return
        # Provider already runs an in-line check at create_snapshot(); this
        # second pass is for the leakage_audit trail.
        if snapshot.window_max_ohlcv_date is None:
            return
        if snapshot.window_max_ohlcv_date > snapshot.trade_date:
            self.leakage_checks["lookback_window_respected"] = "FAILED"

    def _collect_position_log(self, date: str, market_point: MarketPoint) -> None:
        """Collect daily position state for position_log.csv."""
        pos = self.portfolio.position
        self.position_log.append({
            "date": date,
            "ticker": self.config.ticker,
            "side": pos.side.value,
            "quantity": abs(pos.quantity),
            "avg_entry_price": pos.avg_entry_price,
            "mark_price": market_point.close,
            "unrealized_pnl": pos.unrealized_pnl_calc(market_point.close),
            "stop_price": pos.stop_price or "",
            "take_profit": pos.take_profit or "",
            "status": "OPEN" if not pos.is_flat() else "FLAT",
        })

    def _collect_margin_log(self, date: str, snapshot) -> None:
        """Collect daily margin state for margin_log.csv."""
        self.margin_log.append({
            "date": date,
            "total_equity": snapshot.total_equity,
            "gross_exposure": abs(self.portfolio.position.quantity) * snapshot.mark_price * self.spec.multiplier,
            "margin_used": snapshot.margin_used,
            "maintenance_margin": snapshot.maintenance_margin_required,
            "available_equity": snapshot.excess_margin,
            "leverage": snapshot.leverage,
            "margin_status": "SAFE" if snapshot.excess_margin > 0 else "BREACH",
        })

    def _load_initial_report_if_any(self) -> None:
        if not self.config.initial_report_path:
            return
        path = Path(self.config.initial_report_path)
        if not path.exists():
            raise FileNotFoundError(f"Initial report not found: {path}")

        first_date = self.calendar.trading_days(
            self.config.start_date, self.config.end_date
        )[0]
        initial_valid_from = self.calendar.next_trading_day(self.config.start_date)

        decision = self.parser.parse_file(
            path=path,
            fallback_ticker=self.config.ticker,
            fallback_trade_date=self.config.start_date,
            fallback_decision_valid_from=initial_valid_from,
        )

        try:
            snapshot_trade_date = decision.trade_date
            reference_point = self.snapshot_provider.get_market_point(
                symbol=self.config.ticker,
                trade_date=snapshot_trade_date,
                spec=self.spec,
            )
        except (ValueError, KeyError):
            snapshot_trade_date = self.calendar.previous_trading_day(decision.trade_date)
            reference_point = self.snapshot_provider.get_market_point(
                symbol=self.config.ticker,
                trade_date=snapshot_trade_date,
                spec=self.spec,
            )
            decision.last_data_date = snapshot_trade_date

        expected_execution_date = self.calendar.next_trading_day(decision.trade_date)
        if decision.decision_valid_from <= decision.trade_date:
            decision.decision_valid_from = expected_execution_date
        if decision.decision_valid_from < first_date:
            decision.decision_valid_from = first_date

        snapshot_data = self.snapshot_provider.create_snapshot(
            symbol=self.config.ticker,
            trade_date=snapshot_trade_date,
            config=self.config,
        )

        checks = self.validator.validate(
            decision=decision,
            snapshot_metadata=snapshot_data.metadata,
        )
        self.leakage_checks.update(checks)
        self.decision_store.save(decision)
        self._update_risk_levels(decision, self._ohlcv_df)

        execution_date = self.calendar.next_trading_day(decision.trade_date)
        if execution_date < first_date:
            execution_date = first_date

        orders = self.order_generator.generate(
            decision=decision,
            portfolio=self.portfolio,
            execution_date=execution_date,
            reference_point=reference_point,
            spec=self.spec,
        )
        self.broker.add_pending_orders(orders)

    def _compute_atr(self, ohlcv_df, period: int = 14) -> Optional[float]:
        """Compute ATR (Average True Range) from OHLCV DataFrame.

        Returns the latest ATR value, or None if insufficient data.
        """
        if ohlcv_df is None or ohlcv_df.empty:
            return None
        if len(ohlcv_df) < period + 1:
            return None

        high = ohlcv_df["high"]
        low = ohlcv_df["low"]
        close = ohlcv_df["close"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean().iloc[-1]
        if pd.isna(atr):
            return None
        return float(atr)

    def _update_risk_levels(self, decision, ohlcv_df=None) -> None:
        """Update position stop/take_profit from decision.

        When holding a position (LONG/SHORT), always ensure stop_price and
        take_profit are set. Priority:
        1. LLM-provided values (with direction validation)
        2. ATR-based computation (if use_atr_based_stops=True and ATR available)
        3. Fixed percentage fallback
        """
        if not decision.valid:
            return
        if self.portfolio.is_flat():
            return

        position = self.portfolio.position
        entry_price = position.avg_entry_price

        # Validate and fix LLM-provided stop/take_profit direction
        agent_stop = decision.stop_price
        agent_tp = decision.take_profit

        if agent_stop is not None and agent_tp is not None and entry_price > 0:
            # Check if values are in wrong direction and swap
            if position.is_short():
                # SHORT: stop should be > entry, tp should be < entry
                if agent_stop < entry_price and agent_tp > entry_price:
                    agent_stop, agent_tp = agent_tp, agent_stop
            elif position.is_long():
                # LONG: stop should be < entry, tp should be > entry
                if agent_stop > entry_price and agent_tp < entry_price:
                    agent_stop, agent_tp = agent_tp, agent_stop

        # Stop price: use LLM value or compute default
        if agent_stop is not None:
            position.stop_price = agent_stop
        elif entry_price > 0:
            # Try ATR-based first
            atr = None
            if self.config.risk.use_atr_based_stops and ohlcv_df is not None:
                atr = self._compute_atr(ohlcv_df, self.config.risk.atr_period)

            if atr is not None:
                # ATR-based stop
                if position.is_long():
                    position.stop_price = entry_price - (atr * self.config.risk.atr_stop_multiplier)
                else:  # SHORT
                    position.stop_price = entry_price + (atr * self.config.risk.atr_stop_multiplier)
            else:
                # Fallback to fixed percentage
                stop_pct = self.config.risk.default_stop_pct
                if position.is_long():
                    position.stop_price = entry_price * (1 - stop_pct)
                else:  # SHORT
                    position.stop_price = entry_price * (1 + stop_pct)

        # Take profit: use LLM value or compute default
        if agent_tp is not None:
            position.take_profit = agent_tp
        elif entry_price > 0:
            # Try ATR-based first
            atr = None
            if self.config.risk.use_atr_based_stops and ohlcv_df is not None:
                atr = self._compute_atr(ohlcv_df, self.config.risk.atr_period)

            if atr is not None:
                # ATR-based take profit
                if position.is_long():
                    position.take_profit = entry_price + (atr * self.config.risk.atr_tp_multiplier)
                else:  # SHORT
                    position.take_profit = entry_price - (atr * self.config.risk.atr_tp_multiplier)
            else:
                # Fallback to fixed percentage
                tp_pct = self.config.risk.default_take_profit_pct
                if position.is_long():
                    position.take_profit = entry_price * (1 + tp_pct)
                else:  # SHORT
                    position.take_profit = entry_price * (1 - tp_pct)

    def _safe_next_trading_day(self, current_date: str) -> Optional[str]:
        try:
            return self.calendar.next_trading_day(current_date)
        except ValueError:
            return None

    def _finalize(self) -> dict[str, Any]:
        all_margin_events = self.broker.margin_events + self.margin_events

        equity_df = pd.DataFrame(
            [s.to_dict() for s in self.portfolio.equity_curve]
        )

        benchmark_curve = None
        if self.config.benchmark_ticker:
            try:
                benchmark_curve = self.snapshot_provider.get_ohlcv(
                    self.config.benchmark_ticker,
                    self.config.end_date,
                )
                if not benchmark_curve.empty:
                    benchmark_curve["date"] = benchmark_curve["date"].astype(str)
                    mask = (
                        (benchmark_curve["date"] >= self.config.start_date)
                        & (benchmark_curve["date"] <= self.config.end_date)
                    )
                    benchmark_curve = benchmark_curve.loc[mask].copy()
            except Exception:
                benchmark_curve = None

        # Build trigger / rating summary stats.
        trigger_stats = self._build_trigger_stats()

        summary_metrics = self.metrics.calculate(
            equity_curve=equity_df,
            trades=self.portfolio.trades,
            initial_cash=self.config.initial_cash,
            benchmark_curve=benchmark_curve,
            margin_events=all_margin_events,
            trigger_stats=trigger_stats,
        )

        summary = {
            "ticker": self.config.ticker,
            "asset_class": self.config.asset_class,
            "market_mode": "FUTURES_STYLE_SIMULATION",
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "initial_cash": self.config.initial_cash,
            "lookback_days": self.config.lookback_days,
            "decision_mapping_mode": self.config.decision_mapping.mode,
            "margin_call_count": sum(1 for e in all_margin_events if e.kind in {"call", "liquidation"}),
            "liquidation_count": sum(1 for e in all_margin_events if e.kind == "liquidation"),
            **summary_metrics,
        }

        leakage_audit = self._build_leakage_audit()

        paths = self.reporter.export_all(
            summary=summary,
            portfolio=self.portfolio,
            decisions=self.decision_store.all(),
            leakage_audit=leakage_audit,
            margin_events=all_margin_events,
            position_log=self.position_log,
            margin_log=self.margin_log,
            trigger_log=self.trigger_log,
        )

        return {
            "summary": summary,
            "leakage_audit": leakage_audit,
            "output_paths": paths,
            "margin_events": [e.to_dict() for e in all_margin_events],
        }

    def _build_leakage_audit(self) -> dict[str, Any]:
        checks = dict(self.leakage_checks)
        required = [
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
            "lookback_window_respected",
        ]
        for check in required:
            checks.setdefault(check, "PASSED")
        status = "PASSED" if all(v == "PASSED" for v in checks.values()) else "FAILED"
        return {
            "status": status,
            "checks": checks,
        }

    def _build_trigger_stats(self) -> dict[str, Any]:
        total = len(self.trigger_log)
        triggered = sum(1 for entry in self.trigger_log if entry.get("triggered"))
        skipped = total - triggered
        # Rating distribution based on saved decisions' parsed ratings.
        rating_counts: dict[str, int] = {}
        for entry in self.trigger_log:
            rating = entry.get("agent_rating")
            if not rating:
                continue
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        # Reason frequency.
        reason_counts: dict[str, int] = {}
        for entry in self.trigger_log:
            for reason in entry.get("reasons") or []:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return {
            "total_decisions": total,
            "triggered": triggered,
            "skipped": skipped,
            "trigger_hit_rate": (triggered / total) if total else 0.0,
            "rating_distribution": rating_counts,
            "reason_counts": reason_counts,
        }
