"""
Integration tests for 4-analyst backtesting — verifies that the
agent_runner correctly configures 4 analysts and snapshot data.
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tradingagents.backtesting.agent_runner import TradingAgentsRunner
from tradingagents.backtesting.decision_schema import AgentConfig


class Test4AnalystsConfig(unittest.TestCase):
    """Agent runner configures 4 analysts for backtesting."""

    def test_selected_analysts_includes_all_4(self):
        """_build_tradingagents_graph should use 4 analyst keys."""
        runner = TradingAgentsRunner(
            reports_root="/tmp/test_reports",
            agent_config=AgentConfig(backtest_mode=True),
        )
        # Read the source to verify selected_analysts
        import inspect
        source = inspect.getsource(runner._build_tradingagents_graph)
        self.assertIn('"market"', source)
        self.assertIn('"news"', source)
        self.assertIn('"social"', source)
        self.assertIn('"fundamentals"', source)

    def test_data_vendors_set_to_snapshot(self):
        """_safe_agent_runtime_config should set data_vendors to snapshot."""
        runner = TradingAgentsRunner(
            reports_root="/tmp/test_reports",
            agent_config=AgentConfig(backtest_mode=True),
        )
        config = runner._safe_agent_runtime_config()
        data_vendors = config.get("data_vendors", {})
        self.assertEqual(data_vendors.get("core_stock_apis"), "snapshot")
        self.assertEqual(data_vendors.get("technical_indicators"), "snapshot")
        self.assertEqual(data_vendors.get("fundamental_data"), "snapshot")
        self.assertEqual(data_vendors.get("news_data"), "snapshot")

    def test_snapshot_data_injected_when_provided(self):
        """snapshot_data should be in config when snapshot is provided."""
        runner = TradingAgentsRunner(
            reports_root="/tmp/test_reports",
            agent_config=AgentConfig(backtest_mode=True),
        )
        mock_snapshot = MagicMock()
        mock_snapshot.root_path = Path("/tmp/snapshots/TEST/2025-06-14")
        mock_snapshot.spec = MagicMock()
        mock_snapshot.spec.multiplier = 1.0
        mock_snapshot.spec.tick_size = 0.01
        mock_snapshot.ohlcv = MagicMock()
        mock_snapshot.news = [{"title": "test"}]
        mock_snapshot.fundamentals = []
        mock_snapshot.sentiment = []
        mock_snapshot.broker_activity = []

        config = runner._safe_agent_runtime_config(snapshot=mock_snapshot)
        self.assertIn("snapshot_data", config)
        self.assertEqual(config["snapshot_data"]["news"], [{"title": "test"}])

    def test_assert_runtime_config_validates_data_vendors(self):
        """_assert_runtime_config_is_safe should reject non-snapshot vendors."""
        runner = TradingAgentsRunner(
            reports_root="/tmp/test_reports",
            agent_config=AgentConfig(backtest_mode=True),
        )
        # Build a minimal valid config (mimicking what _call_tradingagents_repo does)
        config = runner._safe_agent_runtime_config()
        config["snapshot_path"] = "/tmp/snapshots/TEST/2025-06-14"
        config["data_path"] = "/tmp/snapshots/TEST/2025-06-14"
        config["ticker"] = "TEST"
        config["curr_date"] = "2025-06-14"
        config["trade_date"] = "2025-06-14"

        # Should pass with snapshot vendors
        runner._assert_runtime_config_is_safe(config)

        # Should fail if we change a vendor
        config["data_vendors"]["news_data"] = "yfinance"
        with self.assertRaises(ValueError):
            runner._assert_runtime_config_is_safe(config)


class TestCallbackNodes(unittest.TestCase):
    """Agent callbacks track all 4 analyst nodes."""

    def test_known_nodes_includes_all_analysts(self):
        from tradingagents.backtesting.agent_callbacks import KNOWN_NODES
        self.assertIn("Market Analyst", KNOWN_NODES)
        self.assertIn("News Analyst", KNOWN_NODES)
        self.assertIn("Sentiment Analyst", KNOWN_NODES)
        self.assertIn("Fundamentals Analyst", KNOWN_NODES)


if __name__ == "__main__":
    unittest.main()
