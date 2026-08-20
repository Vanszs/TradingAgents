"""
Performance metrics for the stock margin backtester.

Standard return/risk metrics plus margin-specific:
- average / max leverage
- average / max margin utilization
- margin call count
- liquidation count
- average holding period
- consecutive wins / losses
- long / short realized PnL breakdown
"""
from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from .decision_schema import MarginEvent, OrderSide, Trade


class MetricsCalculator:
    def calculate(
        self,
        equity_curve: pd.DataFrame,
        trades: list[Trade],
        initial_cash: float,
        benchmark_curve: Optional[pd.DataFrame] = None,
        margin_events: Optional[list[MarginEvent]] = None,
        trigger_stats: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        margin_events = margin_events or []
        trigger_stats = trigger_stats or {}

        empty = self._empty_metrics(initial_cash)
        empty.update(self._trigger_metrics_block(trigger_stats))
        if equity_curve.empty:
            return self._augment_with_margin_metrics(empty, trades, margin_events)

        df = equity_curve.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["total_equity"] = pd.to_numeric(df["total_equity"], errors="coerce")
        df["drawdown"] = pd.to_numeric(df["drawdown"], errors="coerce")
        if "position_qty" in df.columns:
            df["position_qty"] = pd.to_numeric(df["position_qty"], errors="coerce").fillna(0)
        else:
            df["position_qty"] = 0

        for col in (
            "margin_used",
            "leverage",
            "daily_realized_pnl",
            "daily_unrealized_pnl",
            "lifetime_realized_pnl",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        df = df.dropna(subset=["date", "total_equity"]).sort_values("date")
        if df.empty:
            return self._augment_with_margin_metrics(empty, trades, margin_events)

        returns: pd.Series = (
            df["total_equity"].pct_change().fillna(0.0).astype(float)
        )
        # Handle edge cases in returns (e.g. equity drop to 0 is -1.0, not 0.0)
        # pct_change() from positive to 0 produces -1.0; 0 to 0 is NaN -> fillna(0.0)
        # from positive to negative or 0 to negative can produce -inf or large negative.
        # Clamp -inf to -1.0 (100% loss) and +inf to 0.0.
        returns = returns.replace([float("-inf")], -1.0)
        returns = returns.replace([float("inf")], 0.0)

        final_equity = float(df["total_equity"].iloc[-1])
        total_return = (
            (final_equity / initial_cash) - 1.0 if initial_cash > 0 else 0.0
        )
        start_date = pd.Timestamp(df["date"].iloc[0])
        end_date = pd.Timestamp(df["date"].iloc[-1])
        days = max((end_date - start_date).days, 1)
        years = days / 365.25
        if final_equity <= 0:
            cagr = -1.0
        elif years > 0 and initial_cash > 0:
            cagr = (final_equity / initial_cash) ** (1 / years) - 1
        else:
            cagr = 0.0

        max_drawdown = float(df["drawdown"].min())
        daily_mean = float(returns.mean())
        daily_std = float(returns.std(ddof=0))
        sharpe = 0.0
        if daily_std > 0:
            sharpe = (daily_mean / daily_std) * math.sqrt(252)
        # True Downside Semi-Deviation (root-mean-square of negative returns relative to 0 across ALL N days)
        downside_diff = returns.clip(upper=0.0).astype(float)
        downside_dev = math.sqrt(float((downside_diff ** 2).mean()))
        sortino = 0.0
        if downside_dev > 0:
            sortino = (daily_mean / downside_dev) * math.sqrt(252)
        elif daily_mean > 0:
            sortino = float("inf")

        trade_stats = self._trade_stats(trades)
        exposure_time = float((df["position_qty"] != 0).mean())

        benchmark_return_pct = None
        alpha_pct = None
        if benchmark_curve is not None and not benchmark_curve.empty:
            br = self._calculate_benchmark_return(benchmark_curve)
            benchmark_return_pct = br
            if br is not None:
                alpha_pct = (total_return * 100) - br

        avg_leverage = float(df["leverage"].mean())
        max_leverage = float(df["leverage"].max())
        avg_margin_util = float(
            (df["margin_used"] / df["total_equity"].replace(0, pd.NA))
            .dropna()
            .astype(float)
            .mean()
            * 100
        ) if (df["margin_used"] != 0).any() else 0.0
        max_margin_util = float(
            (df["margin_used"] / df["total_equity"].replace(0, pd.NA))
            .dropna()
            .astype(float)
            .max()
            * 100
        ) if (df["margin_used"] != 0).any() else 0.0

        base = {
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": (cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
            "win_rate": trade_stats["win_rate"],
            "profit_factor": trade_stats["profit_factor"],
            "average_gain": trade_stats["average_gain"],
            "average_loss": trade_stats["average_loss"],
            "number_of_trades": len(trades),
            "long_win_rate": trade_stats["long_win_rate"],
            "short_win_rate": trade_stats["short_win_rate"],
            "long_trade_count": trade_stats["long_trade_count"],
            "short_trade_count": trade_stats["short_trade_count"],
            "exposure_time_pct": exposure_time * 100,
            "benchmark_return_pct": benchmark_return_pct,
            "alpha_pct": alpha_pct,
            "avg_leverage": avg_leverage,
            "max_leverage": max_leverage,
            "avg_margin_utilization_pct": avg_margin_util,
            "max_margin_utilization_pct": max_margin_util,
            "margin_calls_count": sum(1 for e in margin_events if e.kind in {"call", "liquidation"}),
            "liquidations_count": sum(1 for e in margin_events if e.kind == "liquidation"),
            "reverse_count": sum(1 for e in margin_events if e.kind == "reverse"),
            "avg_holding_period_days": trade_stats["avg_holding_period_days"],
            "consecutive_wins_max": trade_stats["consecutive_wins_max"],
            "consecutive_losses_max": trade_stats["consecutive_losses_max"],
            "long_realized_pnl": trade_stats["long_realized_pnl"],
            "short_realized_pnl": trade_stats["short_realized_pnl"],
            "total_realized_pnl": trade_stats["total_realized_pnl"],
            "total_fees": trade_stats["total_fees"],
            "fee_drag_pct": (trade_stats["total_fees"] / initial_cash * 100) if initial_cash > 0 else 0.0,
            "total_slippage": trade_stats["total_slippage"],
            "slippage_drag_pct": (trade_stats["total_slippage"] / initial_cash * 100) if initial_cash > 0 else 0.0,
            "turnover_notional": trade_stats["turnover_notional"],
        }
        base.update(self._trigger_metrics_block(trigger_stats))
        return base

    @staticmethod
    def _trigger_metrics_block(trigger_stats: dict[str, Any]) -> dict[str, Any]:
        """Project trigger / rating stats into the summary metrics dict."""
        return {
            "trigger_total_decisions": int(trigger_stats.get("total_decisions", 0)),
            "trigger_triggered": int(trigger_stats.get("triggered", 0)),
            "trigger_skipped": int(trigger_stats.get("skipped", 0)),
            "trigger_hit_rate": float(trigger_stats.get("trigger_hit_rate", 0.0)),
            "trigger_reason_counts": dict(trigger_stats.get("reason_counts", {})),
            "rating_distribution": dict(trigger_stats.get("rating_distribution", {})),
        }

    def _trade_stats(self, trades: list[Trade]) -> dict[str, float]:
        long_lots: list[tuple[int, float, float, Optional[str]]] = []
        short_lots: list[tuple[int, float, float, Optional[str]]] = []
        realized_pnls: list[float] = []
        long_pnl_total = 0.0
        short_pnl_total = 0.0
        total_fees = 0.0
        total_slippage = 0.0
        turnover = 0.0
        holding_periods: list[int] = []
        long_holding_periods: list[int] = []
        short_holding_periods: list[int] = []
        long_wins = 0
        long_total = 0
        short_wins = 0
        short_total = 0
        reverse_count = 0

        for trade in trades:
            total_fees += trade.fee
            turnover += trade.notional
            total_slippage += trade.slippage_ticks * trade.tick_size * trade.quantity

            # Normalize side and open_close to strings for robust comparison
            side_str = trade.side.value if hasattr(trade.side, 'value') else str(trade.side)
            oc_str = trade.open_close.value if hasattr(trade.open_close, 'value') else str(trade.open_close)

            # Determine whether this trade opens or closes lots
            if side_str == "BUY":
                if oc_str == "CLOSE" or (oc_str == "AUTO" and len(short_lots) > 0):
                    is_close, is_long = True, False
                else:
                    is_close, is_long = False, True
            elif side_str == "SELL":
                if oc_str == "CLOSE" or (oc_str == "AUTO" and len(long_lots) > 0):
                    is_close, is_long = True, True
                else:
                    is_close, is_long = False, False
            else:
                continue

            if not is_close:
                if is_long:
                    long_lots.append((trade.quantity, trade.price, trade.multiplier, trade.date))
                else:
                    short_lots.append((trade.quantity, trade.price, trade.multiplier, trade.date))
            else:
                qty_to_close = trade.quantity
                close_price = trade.price
                lots = long_lots if is_long else short_lots

                while qty_to_close > 0 and lots:
                    lot_qty, lot_price, lot_mult, entry_date = lots.pop(0)
                    matched = min(qty_to_close, lot_qty)
                    if is_long:
                        pnl = (close_price - lot_price) * matched * lot_mult
                    else:
                        pnl = (lot_price - close_price) * matched * lot_mult
                    realized_pnls.append(pnl)
                    if is_long:
                        long_pnl_total += pnl
                        long_total += 1
                        if pnl > 0:
                            long_wins += 1
                    else:
                        short_pnl_total += pnl
                        short_total += 1
                        if pnl > 0:
                            short_wins += 1

                    if trade.date and entry_date:
                        try:
                            from datetime import datetime
                            d_open = datetime.fromisoformat(str(entry_date)[:10])
                            d_close = datetime.fromisoformat(str(trade.date)[:10])
                            days = max(0, (d_close - d_open).days)
                            holding_periods.append(days)
                            if is_long:
                                long_holding_periods.append(days)
                            else:
                                short_holding_periods.append(days)
                        except Exception:
                            pass

                    remaining = lot_qty - matched
                    if remaining > 0:
                        lots.insert(0, (remaining, lot_price, lot_mult, entry_date))
                    qty_to_close -= matched

                # If closing order exceeded existing lots (position reversal), add residual to opposing inventory
                if qty_to_close > 0:
                    if is_long:  # Excess SELL becomes a short open lot
                        short_lots.append((qty_to_close, close_price, trade.multiplier, trade.date))
                    else:        # Excess BUY becomes a long open lot
                        long_lots.append((qty_to_close, close_price, trade.multiplier, trade.date))

        wins = [p for p in realized_pnls if p > 0]
        losses = [p for p in realized_pnls if p < 0]
        win_rate = (len(wins) / len(realized_pnls) * 100) if realized_pnls else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        average_gain = (sum(wins) / len(wins)) if wins else 0.0
        average_loss = (sum(losses) / len(losses)) if losses else 0.0

        consecutive_wins = self._max_consecutive(realized_pnls, positive=True)
        consecutive_losses = self._max_consecutive(realized_pnls, positive=False)
        avg_holding = (sum(holding_periods) / len(holding_periods)) if holding_periods else 0.0
        avg_long_holding = (
            sum(long_holding_periods) / len(long_holding_periods)
            if long_holding_periods else 0.0
        )
        avg_short_holding = (
            sum(short_holding_periods) / len(short_holding_periods)
            if short_holding_periods else 0.0
        )

        return {
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "average_gain": float(average_gain),
            "average_loss": float(average_loss),
            "avg_holding_period_days": float(avg_holding),
            "avg_holding_period_long_days": float(avg_long_holding),
            "avg_holding_period_short_days": float(avg_short_holding),
            "consecutive_wins_max": int(consecutive_wins),
            "consecutive_losses_max": int(consecutive_losses),
            "long_realized_pnl": float(long_pnl_total),
            "short_realized_pnl": float(short_pnl_total),
            "total_realized_pnl": float(long_pnl_total + short_pnl_total),
            "total_fees": float(total_fees),
            "total_slippage": float(total_slippage),
            "turnover_notional": float(turnover),
            "long_win_rate": float((long_wins / long_total * 100) if long_total > 0 else 0.0),
            "short_win_rate": float((short_wins / short_total * 100) if short_total > 0 else 0.0),
            "long_trade_count": int(long_total),
            "short_trade_count": int(short_total),
        }

    @staticmethod
    def _max_consecutive(pnls: list[float], positive: bool) -> int:
        best = 0
        current = 0
        for p in pnls:
            ok = (p > 0) if positive else (p < 0)
            if ok:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

    def _empty_metrics(self, initial_cash: float) -> dict[str, Any]:
        return {
            "final_equity": initial_cash,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_gain": 0.0,
            "average_loss": 0.0,
            "number_of_trades": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "exposure_time_pct": 0.0,
            "benchmark_return_pct": None,
            "alpha_pct": None,
            "avg_leverage": 0.0,
            "max_leverage": 0.0,
            "avg_margin_utilization_pct": 0.0,
            "max_margin_utilization_pct": 0.0,
            "margin_calls_count": 0,
            "liquidations_count": 0,
            "reverse_count": 0,
            "avg_holding_period_days": 0.0,
            "avg_holding_period_long_days": 0.0,
            "avg_holding_period_short_days": 0.0,
            "consecutive_wins_max": 0,
            "consecutive_losses_max": 0,
            "long_realized_pnl": 0.0,
            "short_realized_pnl": 0.0,
            "total_realized_pnl": 0.0,
            "total_fees": 0.0,
            "fee_drag_pct": 0.0,
            "total_slippage": 0.0,
            "slippage_drag_pct": 0.0,
            "turnover_notional": 0.0,
            "trigger_total_decisions": 0,
            "trigger_triggered": 0,
            "trigger_skipped": 0,
            "trigger_hit_rate": 0.0,
            "trigger_reason_counts": {},
            "rating_distribution": {},
        }

    def _augment_with_margin_metrics(
        self,
        base: dict[str, Any],
        trades: list[Trade],
        margin_events: list[MarginEvent],
    ) -> dict[str, Any]:
        trade_stats = self._trade_stats(trades)
        base.update(
            {
                "avg_leverage": 0.0,
                "max_leverage": 0.0,
                "avg_margin_utilization_pct": 0.0,
                "max_margin_utilization_pct": 0.0,
                "margin_calls_count": sum(1 for e in margin_events if e.kind in {"call", "liquidation"}),
                "liquidations_count": sum(1 for e in margin_events if e.kind == "liquidation"),
                "reverse_count": sum(1 for e in margin_events if e.kind == "reverse"),
                "avg_holding_period_days": trade_stats["avg_holding_period_days"],
                "consecutive_wins_max": trade_stats["consecutive_wins_max"],
                "consecutive_losses_max": trade_stats["consecutive_losses_max"],
                "long_realized_pnl": trade_stats["long_realized_pnl"],
                "short_realized_pnl": trade_stats["short_realized_pnl"],
                "total_realized_pnl": trade_stats["total_realized_pnl"],
                "total_fees": trade_stats["total_fees"],
                "total_slippage": trade_stats["total_slippage"],
                "long_win_rate": trade_stats["long_win_rate"],
                "short_win_rate": trade_stats["short_win_rate"],
                "long_trade_count": trade_stats["long_trade_count"],
                "short_trade_count": trade_stats["short_trade_count"],
                "turnover_notional": trade_stats["turnover_notional"],
            }
        )
        return base

    def _calculate_benchmark_return(
        self,
        benchmark_curve: pd.DataFrame,
    ) -> Optional[float]:
        bdf = benchmark_curve.copy()
        if "date" not in bdf.columns or "close" not in bdf.columns:
            return None
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf["close"] = pd.to_numeric(bdf["close"], errors="coerce")
        bdf = bdf.dropna(subset=["date", "close"]).sort_values("date")
        if len(bdf) < 2:
            return None
        start_price = float(bdf["close"].iloc[0])
        end_price = float(bdf["close"].iloc[-1])
        if start_price <= 0:
            return None
        return ((end_price / start_price) - 1.0) * 100
