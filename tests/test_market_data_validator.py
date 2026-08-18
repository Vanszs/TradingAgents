import unittest
from unittest.mock import patch
import pandas as pd

from tradingagents.dataflows.market_data_validator import validate_market_data, format_verified_market_snapshot


class TestMarketDataValidator(unittest.TestCase):
    def test_validate_market_data_with_mock_ohlcv(self):
        mock_df = pd.DataFrame([
            {"Date": "2025-01-02", "Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 102.0, "Volume": 10000},
            {"Date": "2025-01-03", "Open": 102.0, "High": 108.0, "Low": 101.0, "Close": 107.0, "Volume": 15000},
        ])

        with patch("tradingagents.dataflows.market_data_validator.load_ohlcv", return_value=mock_df):
            snapshot = validate_market_data("BBRI.JK", "2025-01-03")
            self.assertEqual(snapshot["status"], "VERIFIED")
            self.assertEqual(snapshot["trade_date"], "2025-01-03")
            self.assertEqual(snapshot["latest_ohlcv"]["close"], 107.0)

            formatted = format_verified_market_snapshot("BBRI.JK", "2025-01-03")
            self.assertIn("Verified Market Snapshot: BBRI.JK", formatted)
            self.assertIn("Close**: 107.0", formatted)

    def test_validate_market_data_unavailable(self):
        with patch("tradingagents.dataflows.market_data_validator.load_ohlcv", side_effect=Exception("network error")):
            snapshot = validate_market_data("UNKNOWN", "2025-01-03")
            self.assertEqual(snapshot["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
