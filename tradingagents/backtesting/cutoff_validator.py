"""
Anti-data-leakage validator for the stock backtester.

Checks:
- Decision metadata is complete
- Provider is snapshot-only
- OHLCV cutoff respects trade_date
- News, sentiment, fundamental timestamps respect their respective cutoffs
- Decision is only valid from the next trading session (next-bar execution)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from .decision_schema import ParsedDecision, SnapshotMetadata, parse_date


class LeakageValidationError(Exception):
    pass


class DecisionCutoffValidator:
    def __init__(
        self,
        fail_on_future_data: bool = True,
        fundamental_buffer_days: int = 3,
    ):
        self.fail_on_future_data = fail_on_future_data
        self.fundamental_buffer_days = fundamental_buffer_days
        self.audit_checks: dict[str, str] = {}

    def _fail(self, check_name: str, message: str) -> None:
        self.audit_checks[check_name] = "FAILED"
        if self.fail_on_future_data:
            raise LeakageValidationError(message)

    def _pass(self, check_name: str) -> None:
        self.audit_checks[check_name] = "PASSED"

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def validate(
        self,
        decision: ParsedDecision,
        snapshot_metadata: SnapshotMetadata,
    ) -> dict[str, str]:
        if not decision.valid:
            self._fail("decision_valid", decision.invalid_reason or "Invalid decision.")
            return dict(self.audit_checks)

        self._validate_required_fields(decision, snapshot_metadata)
        self._validate_provider(snapshot_metadata)
        self._validate_ohlcv(decision, snapshot_metadata)
        self._validate_news(decision, snapshot_metadata)
        self._validate_fundamentals(decision, snapshot_metadata)
        self._validate_sentiment(decision, snapshot_metadata)
        self._validate_next_bar_execution(decision)

        return dict(self.audit_checks)

    def _validate_required_fields(
        self,
        decision: ParsedDecision,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        required_decision = [
            decision.ticker,
            decision.trade_date,
            decision.report_generated_at,
            decision.last_data_date,
            decision.decision_valid_from,
        ]
        if not all(required_decision):
            self._fail("required_decision_fields", "Decision metadata is incomplete.")
        else:
            self._pass("required_decision_fields")

        if not snapshot_metadata.trade_date:
            self._fail("snapshot_metadata_required", "Snapshot metadata missing trade_date.")
        else:
            self._pass("snapshot_metadata_required")

    def _validate_provider(self, snapshot_metadata: SnapshotMetadata) -> None:
        if snapshot_metadata.provider_mode != "snapshot":
            self._fail(
                "live_provider_disabled",
                "Provider mode must be snapshot during backtest.",
            )
        else:
            self._pass("live_provider_disabled")

    def _validate_ohlcv(
        self,
        decision: ParsedDecision,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        if not snapshot_metadata.max_ohlcv_date:
            self._fail("ohlcv_cutoff", "Snapshot missing max_ohlcv_date.")
            return

        if parse_date(snapshot_metadata.max_ohlcv_date) > parse_date(decision.trade_date):
            self._fail(
                "ohlcv_cutoff",
                f"Future OHLCV detected: {snapshot_metadata.max_ohlcv_date} > {decision.trade_date}",
            )
        elif parse_date(decision.last_data_date) > parse_date(decision.trade_date):
            self._fail(
                "last_data_date_cutoff",
                f"last_data_date {decision.last_data_date} > trade_date {decision.trade_date}",
            )
        else:
            self._pass("ohlcv_cutoff")
            self._pass("last_data_date_cutoff")

    def _validate_news(
        self,
        decision: ParsedDecision,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        max_news = self._parse_dt(snapshot_metadata.max_news_time)
        if max_news is None:
            self._pass("news_cutoff")
            return
        trade_date = parse_date(decision.trade_date)
        max_news_date = max_news.date()
        if max_news_date >= trade_date:
            self._fail(
                "news_cutoff",
                f"News from same day detected: {max_news_date} >= trade_date {trade_date}. "
                f"News must be from previous trading day to prevent leakage.",
            )
        else:
            self._pass("news_cutoff")

    def _validate_fundamentals(
        self,
        decision: ParsedDecision,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        max_available = snapshot_metadata.max_fundamental_available_date
        if not max_available:
            self._pass("fundamental_available_date")
            return
        trade = parse_date(decision.trade_date)
        avail = parse_date(max_available)
        buffered_date = avail + timedelta(days=self.fundamental_buffer_days)
        if buffered_date > trade:
            self._fail(
                "fundamental_available_date",
                f"Fundamental too recent: available {max_available} + "
                f"{self.fundamental_buffer_days} day buffer = {buffered_date} "
                f"> trade_date {trade}",
            )
        else:
            self._pass("fundamental_available_date")

    def _validate_sentiment(
        self,
        decision: ParsedDecision,
        snapshot_metadata: SnapshotMetadata,
    ) -> None:
        max_sent = self._parse_dt(snapshot_metadata.max_sentiment_time)
        if max_sent is None:
            self._pass("sentiment_cutoff")
            return
        trade_date = parse_date(decision.trade_date)
        max_sent_date = max_sent.date()
        if max_sent_date >= trade_date:
            self._fail(
                "sentiment_cutoff",
                f"Sentiment from same day detected: {max_sent_date} >= trade_date {trade_date}. "
                f"Sentiment must be from previous trading day to prevent leakage.",
            )
        else:
            self._pass("sentiment_cutoff")

    def _validate_next_bar_execution(self, decision: ParsedDecision) -> None:
        if parse_date(decision.decision_valid_from) <= parse_date(decision.trade_date):
            self._fail(
                "next_bar_execution",
                (
                    f"decision_valid_from must be after trade_date. "
                    f"Got {decision.decision_valid_from} <= {decision.trade_date}"
                ),
            )
        else:
            self._pass("next_bar_execution")
