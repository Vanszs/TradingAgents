"""
Report generation for the stock margin backtester.

Exports:
- summary.json
- trade_log.csv
- equity_curve.csv
- position_log.csv
- margin_log.csv
- decision_log.jsonl
- leakage_audit.json
- account_state.csv
- margin_events.csv
- report.md
- config.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .decision_schema import (
    BacktestConfig,
    MarginEvent,
    ParsedDecision,
    Trade,
    ensure_dir,
)
from .portfolio import Portfolio


class BacktestReportGenerator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.output_dir = ensure_dir(
            Path(config.output.output_root) / config.ticker
        )

    def export_all(
        self,
        summary: dict[str, Any],
        portfolio: Portfolio,
        decisions: list[ParsedDecision],
        leakage_audit: dict[str, Any],
        margin_events: list[MarginEvent] | None = None,
        position_log: list[dict] | None = None,
        margin_log: list[dict] | None = None,
        trigger_log: list[dict] | None = None,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        margin_events = margin_events or []

        if self.config.output.save_trade_log:
            paths["trade_log"] = str(self.export_trade_log(portfolio.trades))
        if self.config.output.save_equity_curve:
            paths["equity_curve"] = str(self.export_equity_curve(portfolio))
        if self.config.output.save_decisions:
            paths["decision_log"] = str(self.export_decision_log(decisions))

        paths["summary"] = str(self.export_summary(summary))

        if self.config.output.save_account_state:
            paths["account_state"] = str(self.export_account_state(portfolio))

        if self.config.output.save_margin_events:
            paths["margin_events"] = str(self.export_margin_events(margin_events))

        if self.config.output.save_leakage_audit:
            paths["leakage_audit"] = str(self.export_leakage_audit(leakage_audit))

        if position_log:
            paths["position_log"] = str(self.export_position_log(position_log))
        if margin_log:
            paths["margin_log"] = str(self.export_margin_log(margin_log))
        if trigger_log:
            paths["trigger_log"] = str(self.export_trigger_log(trigger_log))

        paths["report"] = str(self.export_markdown_report(summary, leakage_audit, margin_events))
        self.export_config()
        return paths

    def export_trigger_log(self, trigger_log: list[dict]) -> Path:
        """One row per decision: trigger decision for the day."""
        path = self.output_dir / "trigger_log.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for entry in trigger_log:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return path

    def export_summary(self, summary: dict[str, Any]) -> Path:
        path = self.output_dir / "summary.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        return path

    def export_trade_log(self, trades: list[Trade]) -> Path:
        path = self.output_dir / "trade_log.csv"
        rows = [t.to_dict() for t in trades]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_equity_curve(self, portfolio: Portfolio) -> Path:
        path = self.output_dir / "equity_curve.csv"
        rows = [s.to_dict() for s in portfolio.equity_curve]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_decision_log(self, decisions: list[ParsedDecision]) -> Path:
        path = self.output_dir / "decision_log.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for d in decisions:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False, default=str) + "\n")
        return path

    def export_account_state(self, portfolio: Portfolio) -> Path:
        path = self.output_dir / "account_state.csv"
        rows = [s.to_dict() for s in portfolio.equity_curve]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_margin_events(self, events: list[MarginEvent]) -> Path:
        path = self.output_dir / "margin_events.csv"
        rows = [e.to_dict() for e in events]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def export_leakage_audit(self, audit: dict[str, Any]) -> Path:
        path = self.output_dir / "leakage_audit.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
        return path

    def export_position_log(self, position_log: list[dict]) -> Path:
        """PRD §19.4 — position_log.csv."""
        path = self.output_dir / "position_log.csv"
        if position_log:
            pd.DataFrame(position_log).to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)
        return path

    def export_margin_log(self, margin_log: list[dict]) -> Path:
        """PRD §19.5 — margin_log.csv."""
        path = self.output_dir / "margin_log.csv"
        if margin_log:
            pd.DataFrame(margin_log).to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)
        return path

    def export_config(self) -> Path:
        path = self.output_dir / "config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.config,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda o: getattr(o, "__dict__", str(o)),
            )
        return path

    def export_markdown_report(
        self,
        summary: dict[str, Any],
        leakage_audit: dict[str, Any],
        margin_events: list[MarginEvent],
    ) -> Path:
        path = self.output_dir / "report.md"
        checks = leakage_audit.get("checks", {})
        checks_md = "\n".join(
            f"- `{name}`: **{status}**" for name, status in checks.items()
        )

        margin_section = self._format_margin_section(margin_events)

        content = f"""# Backtest Report - {self.config.ticker} (Stock)

## Period

- Asset Class: `{self.config.asset_class}`
- Ticker: `{self.config.ticker}`
- Start Date: `{self.config.start_date}`
- End Date: `{self.config.end_date}`
- Initial Cash: `{self.config.initial_cash:,.2f}`
- Initial Margin %: `{self.config.margin.initial_margin_pct * 100:.2f}%`
- Maintenance Margin %: `{self.config.margin.maintenance_margin_pct * 100:.2f}%`

## Performance Summary

- Final Equity: `{summary.get("final_equity", 0):,.2f}`
- Total Return: `{summary.get("total_return_pct", 0):.2f}%`
- CAGR: `{summary.get("cagr_pct", 0):.2f}%`
- Max Drawdown: `{summary.get("max_drawdown_pct", 0):.2f}%`
- Sharpe Ratio: `{summary.get("sharpe_ratio", 0):.4f}`
- Sortino Ratio: `{summary.get("sortino_ratio", 0):.4f}`
- Win Rate: `{summary.get("win_rate", 0):.2f}%`
- Profit Factor: `{summary.get("profit_factor", 0)}`
- Number of Trades: `{summary.get("number_of_trades", 0)}`
- Exposure Time: `{summary.get("exposure_time_pct", 0):.2f}%`
- Benchmark Return: `{summary.get("benchmark_return_pct")}`
- Alpha: `{summary.get("alpha_pct")}`

## Margin-Specific Metrics

- Average Leverage: `{summary.get("avg_leverage", 0):.2f}x`
- Max Leverage: `{summary.get("max_leverage", 0):.2f}x`
- Average Margin Utilization: `{summary.get("avg_margin_utilization_pct", 0):.2f}%`
- Max Margin Utilization: `{summary.get("max_margin_utilization_pct", 0):.2f}%`
- Margin Calls: `{summary.get("margin_calls_count", 0)}`
- Liquidations: `{summary.get("liquidations_count", 0)}`
- Avg Holding Period (days): `{summary.get("avg_holding_period_days", 0):.2f}`
- Max Consecutive Wins: `{summary.get("consecutive_wins_max", 0)}`
- Max Consecutive Losses: `{summary.get("consecutive_losses_max", 0)}`
- Realized PnL: `{summary.get("total_realized_pnl", 0):,.2f}`
- Calmar Ratio: `{summary.get("calmar_ratio", 0):.4f}`
- Fee Drag: `{summary.get("fee_drag_pct", 0):.4f}%`
- Slippage Drag: `{summary.get("slippage_drag_pct", 0):.4f}%`
- Total Fees: `{summary.get("total_fees", 0):,.2f}`
- Total Slippage: `{summary.get("total_slippage", 0):,.2f}`
- Turnover Notional: `{summary.get("turnover_notional", 0):,.2f}`

## Margin Activity

{margin_section}

## Leakage Audit

Status: **{leakage_audit.get("status", "UNKNOWN")}**

{checks_md}

## Notes

- Asset class: **stock**, spot long-only. Orders are BUY entries or WNS/no-order; exits use static stop-loss, take-profit, time stop, or end-of-horizon closure.
- Decision pada tanggal `t` hanya dieksekusi pada trading day berikutnya.
- Harga eksekusi memakai `next session open` dengan ``tick_slippage`` ticks.
- Live provider wajib disabled saat backtest historis.
- ``account_state.csv`` berisi snapshot harian (cash, position, margin, leverage).
"""
        path.write_text(content, encoding="utf-8")
        return path

    def _format_margin_section(self, events: list[MarginEvent]) -> str:
        if not events:
            return "No margin events."
        lines = ["| Date | Kind | Mark | Deficit | Account Equity | Action |",
                 "|------|------|------|---------|----------------|--------|"]
        for e in events:
            lines.append(
                f"| {e.date} | {e.kind} | "
                f"{e.mark_price:.2f} | {e.deficit:,.2f} | "
                f"{e.account_equity:,.2f} | {e.action} |"
            )
        return "\n".join(lines)


def build_leakage_audit(leakage_checks: dict[str, str]) -> dict[str, Any]:
    """Helper to assemble standard leakage audit structure."""
    checks = dict(leakage_checks)
    required = [
        "ohlcv_cutoff",
        "news_cutoff",
        "fundamental_available_date",
        "next_bar_execution",
        "memory_isolation",
        "live_provider_disabled",
        "lookback_window_respected",
    ]
    for check in required:
        checks.setdefault(check, "NOT_CHECKED")
    status = "PASSED" if all(v == "PASSED" for v in checks.values()) else "FAILED"
    return {
        "status": status,
        "checks": checks,
    }


def build_trigger_stats(trigger_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Helper to compute aggregate trigger summary statistics."""
    total = len(trigger_log)
    triggered = sum(1 for entry in trigger_log if entry.get("triggered"))
    skipped = total - triggered
    rating_counts: dict[str, int] = {}
    for entry in trigger_log:
        rating = entry.get("agent_rating")
        if not rating:
            continue
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    reason_counts: dict[str, int] = {}
    for entry in trigger_log:
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
