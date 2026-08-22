"""
Trading calendar utilities for futures backtests.

Most futures (equity index, FX, crypto) trade Sun-Fri with a brief daily break,
so a Mon-Fri calendar works as a reasonable proxy when no per-session calendar
is available. The class below accepts an explicit list of market_dates (read
from data) for accuracy.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional


class TradingCalendar:
    """
    Simple trading calendar.

    Behavior:
    - If ``market_dates`` is given, the calendar follows those dates exactly.
    - Otherwise it falls back to Mon-Fri (excludes Saturday and Sunday).
    """

    def __init__(self, market_dates: Optional[list[str]] = None):
        self.market_dates: Optional[list[date]] = None
        if market_dates:
            self.market_dates = sorted(self._to_date(d) for d in market_dates)

    @staticmethod
    def _to_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return datetime.fromisoformat(str(value)[:10]).date()

    def is_trading_day(self, value: str | date) -> bool:
        d = self._to_date(value)
        if self.market_dates is not None:
            return d in self.market_dates
        return d.weekday() < 5

    def trading_days(self, start_date: str | date, end_date: str | date) -> list[str]:
        start = self._to_date(start_date)
        end = self._to_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")

        if self.market_dates is not None:
            return [
                d.isoformat() for d in self.market_dates if start <= d <= end
            ]

        days: list[str] = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current.isoformat())
            current += timedelta(days=1)
        return days

    def next_trading_day(self, value: str | date) -> str:
        d = self._to_date(value) + timedelta(days=1)
        if self.market_dates is not None:
            for market_date in self.market_dates:
                if market_date >= d:
                    return market_date.isoformat()
            raise ValueError(f"No next trading day found after {value}")
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d.isoformat()

    def previous_trading_day(self, value: str | date) -> str:
        d = self._to_date(value) - timedelta(days=1)
        if self.market_dates is not None:
            previous = [md for md in self.market_dates if md <= d]
            if not previous:
                raise ValueError(f"No previous trading day found before {value}")
            return previous[-1].isoformat()
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d.isoformat()

    def trading_days_from_start(
        self, start_date: str | date, n_days: int
    ) -> str:
        """
        Return the date of the Nth trading day from start_date (inclusive).

        Example: start_date=2026-06-07, n_days=5 → 5th trading day from
        2026-06-07 inclusive. If 2026-06-07 is a weekend, the count starts
        from the first trading day on or after that date.
        """
        if n_days <= 0:
            raise ValueError(f"n_days must be > 0, got {n_days}")

        if self.market_dates is not None:
            start = self._to_date(start_date)
            count = 0
            for market_date in self.market_dates:
                if market_date >= start:
                    count += 1
                    if count == n_days:
                        return market_date.isoformat()
            raise ValueError(
                f"Cannot find {n_days} trading days from {start_date}; "
                f"only {count} available in market_dates."
            )

        start = self._to_date(start_date)
        current = start
        count = 0
        while True:
            if self.is_trading_day(current):
                count += 1
                if count == n_days:
                    return current.isoformat()
            current += timedelta(days=1)
            if current - start > timedelta(days=365 * 5):
                raise ValueError(
                    f"Cannot find {n_days} trading days from {start_date} "
                    f"within 5 years."
                )
