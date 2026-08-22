"""
Unit tests for WalkForwardBacktestRunner._update_risk_levels().

Covers:
- LLM-provided stop/TP with wrong direction gets swapped
- ATR-based stop calculation
- Fixed percentage fallback when ATR unavailable
- Flat position → no-op
- Missing stop/TP → default computation
"""
import unittest
from unittest.mock import MagicMock

import pandas as pd

from tradingagents.backtesting.position import (
    BacktestConfig,
    DecisionMappingConfig,
    ExecutionConfig,
    MarginConfig,
    Position,
    PositionSide,
    RiskConfig,
)
from tradingagents.backtesting.walk_forward_runner import WalkForwardBacktestRunner


def _make_runner(**risk_overrides) -> WalkForwardBacktestRunner:
    """Build a minimal runner with a mock agent_runner."""
    risk_cfg = RiskConfig(**risk_overrides) if risk_overrides else RiskConfig()
    cfg = BacktestConfig(
        ticker="AAPL",
        start_date="2024-01-02",
        end_date="2024-01-10",
        initial_cash=100_000.0,
        execution=ExecutionConfig(lot_size=1),
        margin=MarginConfig(),
        risk=risk_cfg,
        decision_mapping=DecisionMappingConfig(mode="strict_5tier"),
    )
    # SnapshotProvider needs real OHLCV — mock it
    runner = WalkForwardBacktestRunner.__new__(WalkForwardBacktestRunner)
    runner.config = cfg
    runner.portfolio = MagicMock()
    runner._ohlcv_df = _make_ohlcv()
    return runner


def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    """Generate synthetic OHLCV for ATR computation."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": [100.0 + i * 0.5 for i in range(n)],
        "high": [102.0 + i * 0.5 for i in range(n)],
        "low": [98.0 + i * 0.5 for i in range(n)],
        "close": [100.0 + i * 0.5 for i in range(n)],
        "volume": [1_000_000] * n,
    })


def _make_decision(stop=None, tp=None, valid=True):
    """Minimal mock decision."""
    d = MagicMock()
    d.valid = valid
    d.stop_price = stop
    d.take_profit = tp
    return d


class TestUpdateRiskLevels(unittest.TestCase):

    def test_flat_position_noop(self):
        """When portfolio is flat, _update_risk_levels does nothing."""
        runner = _make_runner()
        runner.portfolio.is_flat.return_value = True
        pos = Position(ticker="AAPL", quantity=0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=90, tp=120))
        self.assertIsNone(pos.stop_price)
        self.assertIsNone(pos.take_profit)

    def test_invalid_decision_noop(self):
        """Invalid decision should not update risk levels."""
        runner = _make_runner()
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(valid=False))
        self.assertIsNone(pos.stop_price)
        self.assertIsNone(pos.take_profit)

    def test_llm_stop_tp_applied_long(self):
        """LLM-provided stop/TP are applied directly for long positions."""
        runner = _make_runner()
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=95.0, tp=110.0))
        self.assertEqual(pos.stop_price, 95.0)
        self.assertEqual(pos.take_profit, 110.0)

    def test_llm_stop_tp_applied_short(self):
        """LLM-provided stop/TP are applied directly for short positions."""
        runner = _make_runner()
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=-100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=105.0, tp=90.0))
        self.assertEqual(pos.stop_price, 105.0)
        self.assertEqual(pos.take_profit, 90.0)

    def test_llm_wrong_direction_swapped_long(self):
        """If LLM puts stop above entry and TP below entry for long, they swap."""
        runner = _make_runner()
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        # stop=110 (> entry), tp=90 (< entry) → swapped to stop=90, tp=110
        runner._update_risk_levels(_make_decision(stop=110.0, tp=90.0))
        self.assertEqual(pos.stop_price, 90.0)
        self.assertEqual(pos.take_profit, 110.0)

    def test_llm_wrong_direction_swapped_short(self):
        """If LLM puts stop below entry and TP above entry for short, they swap."""
        runner = _make_runner()
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=-100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        # stop=90 (< entry), tp=110 (> entry) → swapped to stop=110, tp=90
        runner._update_risk_levels(_make_decision(stop=90.0, tp=110.0))
        self.assertEqual(pos.stop_price, 110.0)
        self.assertEqual(pos.take_profit, 90.0)

    def test_fixed_pct_fallback_stop_long(self):
        """When no LLM stop and ATR disabled, use fixed percentage stop."""
        runner = _make_runner(use_atr_based_stops=False, default_stop_pct=0.08)
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=None, tp=None))
        # 100 * (1 - 0.08) = 92.0
        self.assertAlmostEqual(pos.stop_price, 92.0)
        # 100 * (1 + 0.20) = 120.0 (default_take_profit_pct=0.20)
        self.assertAlmostEqual(pos.take_profit, 120.0)

    def test_fixed_pct_fallback_stop_short(self):
        """When no LLM stop and ATR disabled, use fixed percentage stop for short."""
        runner = _make_runner(use_atr_based_stops=False, default_stop_pct=0.08)
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=-100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=None, tp=None))
        # 100 * (1 + 0.08) = 108.0
        self.assertAlmostEqual(pos.stop_price, 108.0)
        # 100 * (1 - 0.20) = 80.0
        self.assertAlmostEqual(pos.take_profit, 80.0)

    def test_atr_based_stop_long(self):
        """ATR-based stop uses entry - ATR * multiplier for long."""
        runner = _make_runner(
            use_atr_based_stops=True,
            atr_period=14,
            atr_stop_multiplier=1.5,
            atr_tp_multiplier=2.0,
        )
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=None, tp=None))
        # ATR should be computed from the synthetic OHLCV
        self.assertIsNotNone(pos.stop_price)
        self.assertIsNotNone(pos.take_profit)
        # stop should be below entry, tp above entry
        self.assertLess(pos.stop_price, 100.0)
        self.assertGreater(pos.take_profit, 100.0)

    def test_no_llm_no_ohlcv_uses_fixed_pct(self):
        """When OHLCV is None, falls back to fixed percentage."""
        runner = _make_runner(use_atr_based_stops=True)
        runner._ohlcv_df = None
        runner.portfolio.is_flat.return_value = False
        pos = Position(ticker="AAPL", quantity=100, avg_entry_price=100.0)
        runner.portfolio.position = pos
        runner._update_risk_levels(_make_decision(stop=None, tp=None))
        self.assertAlmostEqual(pos.stop_price, 92.0)  # 8% default


if __name__ == "__main__":
    unittest.main()
