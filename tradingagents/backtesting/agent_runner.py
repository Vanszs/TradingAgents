"""
Adapter for running TradingAgents in a stock backtest loop.

Required safety:
- backtest_mode=True
- memory_enabled=False
- web_search_enabled=False
- all providers = snapshot
- asset_type="stock"
"""
from __future__ import annotations

import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.setup import CANONICAL_ANALYST_ORDER

from .decision_schema import AgentConfig, AssetClass, ensure_dir
from .portfolio import Portfolio
from .snapshot_provider import DataSnapshot

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 20

# Canonical 5-tier rating vocabulary, ordered most-bullish to most-bearish.
_RATING_WORDS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
_RATING_WORDS_LOWER = tuple(w.lower() for w in _RATING_WORDS)
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


def _derive_rating_from_text(text: str) -> Optional[str]:
    """Return the most prominent rating word in ``text`` or ``None``.

    Uses a two-pass heuristic:
      1. Look for an explicit "Rating: X" / "Rating - X" label.
      2. Fall back to the first 5-tier rating word found anywhere.
    """
    if not text:
        return None
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m:
            word = m.group(1).strip("*:., ").capitalize()
            if word in _RATING_WORDS:
                return word
    for word in text.lower().split():
        clean = word.strip("*:.,'\"()[]")
        if clean in _RATING_WORDS_LOWER:
            return clean.capitalize()
    return None


class TradingAgentsRunner:
    def __init__(
        self,
        reports_root: str = "reports",
        agent_config: Optional[AgentConfig] = None,
        run_callable: Optional[Callable[..., str]] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ):
        self.reports_root = Path(reports_root)
        self.agent_config = agent_config or AgentConfig()
        self.run_callable = run_callable
        # progress_callback signature: (phase, trade_date, day_idx,
        # n_days, **extra). When supplied, per-node LLM timings are
        # emitted via a BacktestAgentCallback attached to the graph.
        self._progress = progress_callback

    def run(
        self,
        ticker: str,
        trade_date: str,
        snapshot: DataSnapshot,
        portfolio: Optional[Portfolio] = None,
        *,
        day_idx: int = 0,
        n_days: int = 0,
    ) -> str:
        self._validate_safe_agent_config()

        run_start_time = time.time()
        logger.info(f"[AGENT_RUN] Starting agent run for {ticker} on {trade_date} (day {day_idx}/{n_days})")

        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            attempt_start = time.time()
            logger.info(f"[AGENT_RUN] Attempt {attempt}/{MAX_RETRIES}")
            try:
                if self.run_callable is not None:
                    report = self.run_callable(
                        ticker=ticker,
                        trade_date=trade_date,
                        snapshot_path=str(snapshot.root_path),
                        snapshot=snapshot,
                        config=self._safe_agent_runtime_config(
                            snapshot=snapshot,
                            ticker=ticker,
                            trade_date=trade_date,
                            portfolio=portfolio,
                        ),
                    )
                else:
                    report = self._call_tradingagents_repo(
                        ticker=ticker,
                        trade_date=trade_date,
                        snapshot=snapshot,
                        portfolio=portfolio,
                        day_idx=day_idx,
                        n_days=n_days,
                    )
                attempt_duration = time.time() - attempt_start
                logger.info(f"[AGENT_RUN] Attempt {attempt} SUCCESS in {attempt_duration:.3f}s")
                break
            except Exception as exc:
                attempt_duration = time.time() - attempt_start
                last_exc = exc
                logger.error(f"[AGENT_RUN] Attempt {attempt} FAILED in {attempt_duration:.3f}s: {type(exc).__name__}: {exc}")
                print(
                    f"  [agent_runner] Attempt {attempt}/{MAX_RETRIES} failed: {exc}",
                )
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * attempt
                    logger.info(f"[AGENT_RUN] Retrying in {delay}s...")
                    print(f"  [agent_runner] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"[AGENT_RUN] All {MAX_RETRIES} attempts failed. Last error: {exc}")
                    raise last_exc

        total_duration = time.time() - run_start_time
        report_path = self.save_report(ticker, trade_date, report)
        logger.info(f"[AGENT_RUN] Completed in {total_duration:.3f}s total | report saved to {report_path}")
        return report_path.read_text(encoding="utf-8")

    def save_report(self, ticker: str, trade_date: str, report: str) -> Path:
        report_dir = ensure_dir(self.reports_root / ticker / trade_date)
        report_path = report_dir / "complete_report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def _validate_safe_agent_config(self) -> None:
        if not self.agent_config.backtest_mode:
            raise ValueError("Agent must run with backtest_mode=True.")
        if self.agent_config.memory_enabled:
            raise ValueError("Agent memory must be disabled for backtest.")
        if self.agent_config.web_search_enabled:
            raise ValueError("Agent live web search must be disabled for backtest.")
        if self.agent_config.news_provider != "snapshot":
            raise ValueError("Agent news_provider must be snapshot.")

    def _safe_agent_runtime_config(
        self,
        snapshot: Optional[DataSnapshot] = None,
        ticker: Optional[str] = None,
        trade_date: Optional[str] = None,
        portfolio: Optional[Portfolio] = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = deepcopy(DEFAULT_CONFIG)
        config.update(
            {
                "asset_type": AssetClass.STOCK.value,
                "backtest_mode": True,
                "point_in_time_mode": True,
                "memory_enabled": False,
                "web_search_enabled": False,
                "data_provider": "snapshot",
                "market_data_provider": "snapshot",
                "news_provider": "snapshot",
                "fundamentals_provider": "snapshot",
                "fundamentals_data_provider": "snapshot",
                "sentiment_provider": "snapshot",
                "broker_activity_provider": "snapshot",
                "report_language": self.agent_config.report_language,
                "data_cache_dir": "backtest_cache/data",
                "results_dir": "backtest_cache/results",
                "disable_live_news": True,
                "disable_live_web_search": True,
                "disable_live_fundamentals": True,
                "require_fundamental_available_date": True,
                "data_vendors": {
                    "core_stock_apis": "snapshot",
                    "technical_indicators": "snapshot",
                    "fundamental_data": "snapshot",
                    "news_data": "snapshot",
                },
            }
        )

        if snapshot is not None:
            config.update(
                {
                    "snapshot_path": str(snapshot.root_path),
                    "data_path": str(snapshot.root_path),
                    "snapshot_dir": str(snapshot.root_path),
                    "snapshot_root": str(snapshot.root_path.parent.parent),
                    "instrument_multiplier": snapshot.spec.multiplier,
                    "instrument_tick_size": snapshot.spec.tick_size,
                    "snapshot_data": {
                        "ohlcv": snapshot.ohlcv,
                        "news": snapshot.news,
                        "fundamentals": snapshot.fundamentals,
                        "sentiment": snapshot.sentiment,
                        "broker_activity": snapshot.broker_activity,
                        "spec": snapshot.spec,
                    },
                }
            )

        if ticker is not None:
            config.update(
                {
                    "ticker": ticker,
                    "symbol": ticker,
                    "company_name": ticker,
                }
            )

        if trade_date is not None:
            config.update(
                {
                    "curr_date": trade_date,
                    "trade_date": trade_date,
                    "current_date": trade_date,
                }
            )

        if portfolio is not None:
            mark = portfolio.last_mark or portfolio.position.avg_price
            config.update(
                {
                    "account_cash": portfolio.cash,
                    "account_equity": portfolio.account_equity(mark),
                    "position_qty": portfolio.position.quantity,
                    "position_avg_price": portfolio.position.avg_price,
                    "margin_used": portfolio.margin_used(),
                    "margin_available": portfolio.margin_available(mark),
                    "leverage": portfolio.leverage(mark),
                }
            )

        return config

    def _assert_runtime_config_is_safe(self, config: dict[str, Any]) -> None:
        required_keys = [
            "data_cache_dir",
            "results_dir",
            "llm_provider",
            "deep_think_llm",
            "quick_think_llm",
            "max_debate_rounds",
            "max_risk_discuss_rounds",
            "snapshot_path",
            "data_path",
            "ticker",
            "curr_date",
            "trade_date",
            "asset_type",
        ]
        missing = [
            key for key in required_keys
            if key not in config or config.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Unsafe/incomplete TradingAgents backtest config. "
                f"Missing keys: {missing}."
            )

        if config.get("asset_type") != AssetClass.STOCK.value:
            raise ValueError(
                f"asset_type must be 'stock' for this backtester, "
                f"got {config.get('asset_type')!r}"
            )

        required_values = {
            "backtest_mode": True,
            "point_in_time_mode": True,
            "memory_enabled": False,
            "web_search_enabled": False,
            "disable_live_news": True,
            "disable_live_web_search": True,
            "disable_live_fundamentals": True,
            "require_fundamental_available_date": True,
            "news_provider": "snapshot",
            "fundamentals_provider": "snapshot",
            "sentiment_provider": "snapshot",
            "data_provider": "snapshot",
            "market_data_provider": "snapshot",
            "fundamentals_data_provider": "snapshot",
            "broker_activity_provider": "snapshot",
        }
        for key, expected in required_values.items():
            actual = config.get(key)
            if actual != expected:
                raise ValueError(
                    f"Unsafe backtest config: {key} must be {expected}, got {actual}"
                )

        # Validate data_vendors — all categories must point to snapshot
        data_vendors = config.get("data_vendors", {})
        for category, vendor in data_vendors.items():
            if vendor != "snapshot":
                raise ValueError(
                    f"Unsafe backtest config: data_vendors['{category}'] "
                    f"must be 'snapshot', got '{vendor}'"
                )

    def _call_tradingagents_repo(
        self,
        ticker: str,
        trade_date: str,
        snapshot: DataSnapshot,
        portfolio: Optional[Portfolio] = None,
        *,
        day_idx: int = 0,
        n_days: int = 0,
    ) -> str:
        logger.info(f"[AGENT_CALL] Building TradingAgentsGraph for {ticker} on {trade_date}")
        build_start = time.time()

        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
        except ImportError as exc:
            raise ImportError(
                "Cannot import TradingAgentsGraph from "
                "tradingagents.graph.trading_graph."
            ) from exc

        safe_config = self._safe_agent_runtime_config(
            snapshot=snapshot,
            ticker=ticker,
            trade_date=trade_date,
            portfolio=portfolio,
        )
        safe_config["asset_type"] = AssetClass.STOCK.value
        self._assert_runtime_config_is_safe(safe_config)

        # Per-node LLM timing callback (only when a progress_callback
        # was supplied at construction time). The handler hooks into
        # the graph so each langgraph node reports its duration as
        # ``phase="agent_node"`` events.
        agent_callback = None
        if self._progress is not None:
            from .agent_callbacks import BacktestAgentCallback

            agent_callback = BacktestAgentCallback(
                self._progress,
                trade_date=str(trade_date),
                day_idx=day_idx,
                n_days=n_days,
            )

        graph = self._build_tradingagents_graph(
            TradingAgentsGraph=TradingAgentsGraph,
            safe_config=safe_config,
            agent_callback=agent_callback,
        )
        graph.ticker = ticker
        logger.info(f"[AGENT_CALL] Graph built in {time.time() - build_start:.3f}s")

        logger.info(f"[AGENT_CALL] Running graph._run_graph for {ticker} on {trade_date}")
        run_start = time.time()
        result = graph._run_graph(
            company_name=ticker,
            trade_date=trade_date,
            asset_type=AssetClass.STOCK.value,
        )
        logger.info(f"[AGENT_CALL] Graph execution completed in {time.time() - run_start:.3f}s")

        return self._normalize_graph_result_to_report(
            result=result,
            ticker=ticker,
            trade_date=trade_date,
            snapshot=snapshot,
        )

    @staticmethod
    def _extract_final_state(result: Any) -> dict:
        """Pull the merged final_state dict out of whatever _run_graph returned.

        ``TradingAgentsGraph._run_graph`` returns ``(final_state, signal)``;
        older call sites returned a dict; tests sometimes hand in a plain
        string. Normalise all three shapes to a ``dict`` so the downstream
        report composer can read ``final_trade_decision`` reliably.
        """
        if isinstance(result, tuple):
            if not result:
                return {}
            first = result[0]
            if isinstance(first, dict):
                return first
            return {}
        if isinstance(result, dict):
            return result
        return {}

    def _build_tradingagents_graph(
        self,
        TradingAgentsGraph: Any,
        safe_config: dict[str, Any],
        agent_callback: Optional[Any] = None,
    ) -> Any:
        selected_analysts = list(CANONICAL_ANALYST_ORDER)
        asset_type = safe_config.get("asset_type", "stock")
        # Callback list passed to TradingAgentsGraph; the LLM client
        # constructor forwards it to ChatOpenAI / ChatAnthropic / etc.
        callback_list = [agent_callback] if agent_callback is not None else None
        candidates = [
            lambda: TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=False,
                config=safe_config,
                callbacks=callback_list,
                asset_type=asset_type,
            ),
            lambda: TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=False,
                config=safe_config,
                asset_type=asset_type,
            ),
            lambda: TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=False,
                config=safe_config,
            ),
            lambda: TradingAgentsGraph(
                debug=False,
                config=safe_config,
                asset_type=asset_type,
            ),
            lambda: TradingAgentsGraph(
                debug=False,
                config=safe_config,
            ),
        ]
        last_exc: Optional[Exception] = None
        for ctor in candidates:
            try:
                graph = ctor()
            except TypeError as exc:
                last_exc = exc
                continue
            # Attach the agent callback to the propagator so the
            # graph runtime can route per-node events to it. The
            # graph also receives it via ``callbacks=`` above, which
            # the LLM client consumes; the propagator path covers
            # the node-level (chain) events.
            if agent_callback is not None and hasattr(graph, "propagator"):
                try:
                    graph.propagator.callbacks = [agent_callback]
                except Exception:  # noqa: BLE001
                    pass
            return graph
        raise TypeError(
            "Cannot instantiate TradingAgentsGraph with any known signature."
        ) from last_exc

    def _stringify_graph_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            lines: list[str] = []
            for key, item in value.items():
                lines.append(f"### {key}")
                lines.append("")
                lines.append(self._stringify_graph_value(item))
                lines.append("")
            return "\n".join(lines)
        if isinstance(value, list):
            return "\n".join(
                f"- {self._stringify_graph_value(item)}" for item in value
            )
        return str(value)

    def _normalize_graph_result_to_report(
        self,
        result: Any,
        ticker: str,
        trade_date: str,
        snapshot: DataSnapshot,
    ) -> str:
        if isinstance(result, str):
            report = result
        elif isinstance(result, dict):
            report = (
                result.get("complete_report")
                or result.get("final_report")
                or result.get("report")
                or result.get("investment_plan")
                or result.get("trader_investment_plan")
                or result.get("portfolio_management_decision")
                or result.get("final_trade_decision")
                or result.get("decision")
                or self._stringify_graph_value(result)
            )
        elif isinstance(result, tuple):
            # _run_graph returns (final_state, signal). Extract the
            # Portfolio Manager's decision from final_state so the parser
            # sees the same markdown shape the report writer produces,
            # rather than a stringified copy of the entire state dict.
            final_state = self._extract_final_state(result)
            signal = result[1] if len(result) > 1 else None
            if final_state:
                pm_report = (
                    final_state.get("complete_report")
                    or final_state.get("final_trade_decision")
                    or final_state.get("final_report")
                    or final_state.get("report")
                    or final_state.get("investment_plan")
                    or final_state.get("portfolio_management_decision")
                )
                # Merge PM output (authoritative rating & decision) ahead of trader plan
                trader_report = final_state.get("trader_investment_plan", "")
                if pm_report and trader_report:
                    report = f"{pm_report}\n\n{trader_report}"
                elif pm_report:
                    report = pm_report
                elif trader_report:
                    report = trader_report
                else:
                    report = None
                if not report:
                    # Fall back to composing a minimal report from the
                    # final_trade_decision + a derived rating signal.
                    pm_decision = final_state.get("final_trade_decision", "")
                    if signal and "rating" not in pm_decision.lower():
                        report = (
                            f"**Rating**: {signal}\n\n"
                            f"**Executive Summary**: {pm_decision}"
                        )
                    else:
                        report = pm_decision or self._stringify_graph_value(
                            final_state
                        )
            elif isinstance(signal, str):
                # Some pipelines return (signal, ) only. Promote to a
                # parser-friendly markdown body.
                report = f"**Rating**: {signal}" if signal else ""
            else:
                parts: list[str] = []
                for item in result:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(self._stringify_graph_value(item))
                    else:
                        parts.append(str(item))
                report = "\n\n".join(parts)
        else:
            report = str(result)

        # Guarantee the markdown carries a recognisable rating header so the
        # backtest markdown_parser (which keys off `**Rating**:` / `Rating:`)
        # can always extract at least a rating. Without this guard, an LLM
        # that emits only a one-word rating like "Sell" or "Hold" leaves
        # the parser with nothing structured to work with.
        report = self._ensure_rating_header(report)

        last_data_date = snapshot.metadata.max_ohlcv_date or trade_date
        decision_valid_from = self._next_valid_placeholder(trade_date, getattr(self, '_calendar', None))
        required_markers = [
            "Generated:",
            "Trade Date:",
            "Last Data Date:",
            "Decision Valid From:",
        ]
        has_complete_metadata = all(marker in report for marker in required_markers)
        if has_complete_metadata:
            return report

        return f"""# Trading Analysis Report: {ticker}

Generated: {trade_date} 16:30:00
Trade Date: {trade_date}
Last Data Date: {last_data_date}
Decision Valid From: {decision_valid_from}

## Raw TradingAgents Output

{report}
    """

    @staticmethod
    def _ensure_rating_header(report: str) -> str:
        """Prepend a ``**Rating**: X`` header when the LLM output lacks one.

        The Portfolio Manager's structured path always emits a
        ``**Rating**: ...`` line, but the free-text fallback (and certain
        weak models) can return prose that only contains a single rating
        word like ``"Sell"`` or ``"Buy"``. The backtest markdown_parser
        requires a recognisable rating label, so we wrap the response with
        one whenever it is missing — deriving the rating heuristically.
        """
        if not report:
            return report
        if "**Rating**" in report or re.search(r"(?im)^Rating\s*[:\-]", report):
            return report

        rating = _derive_rating_from_text(report)
        if rating is None:
            return report

        return f"**Rating**: {rating}\n\n{report.strip()}"

    @staticmethod
    def _next_valid_placeholder(trade_date: str, calendar=None) -> str:
        """Return the next valid trading day after trade_date.

        If a TradingCalendar is provided, uses it to skip holidays.
        Otherwise falls back to weekend-only skipping.
        """
        if calendar is not None:
            try:
                next_day = calendar.next_trading_day(trade_date)
                if next_day:
                    return next_day
            except (ValueError, KeyError):
                pass
        # Fallback: skip weekends only
        from datetime import datetime, timedelta

        d = datetime.fromisoformat(trade_date[:10]).date()
        d = d + timedelta(days=1)
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        return d.isoformat()
