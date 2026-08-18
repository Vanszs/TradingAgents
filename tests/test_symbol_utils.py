import unittest

from cli.models import AssetType
from cli.utils import detect_asset_type, normalize_ticker_symbol
from tradingagents.dataflows.symbol_utils import is_yahoo_safe, normalize_symbol, NoMarketDataError


class TestSymbolUtils(unittest.TestCase):
    def test_forex_pairs(self):
        self.assertEqual(normalize_symbol("EURUSD"), "EURUSD=X")
        self.assertEqual(normalize_symbol("GBPJPY"), "GBPJPY=X")
        self.assertEqual(normalize_symbol("USDIDR"), "USDIDR=X")
        self.assertEqual(normalize_symbol("EUR/USD"), "EURUSD=X")

    def test_commodities_and_metals(self):
        self.assertEqual(normalize_symbol("XAUUSD"), "GC=F")
        self.assertEqual(normalize_symbol("XAUUSD+"), "GC=F")
        self.assertEqual(normalize_symbol("GOLD"), "GC=F")
        self.assertEqual(normalize_symbol("SILVER"), "SI=F")
        self.assertEqual(normalize_symbol("USOIL"), "CL=F")
        self.assertEqual(normalize_symbol("WTI"), "CL=F")
        self.assertEqual(normalize_symbol("BRENT"), "BZ=F")

    def test_crypto_aliases(self):
        self.assertEqual(normalize_symbol("BTCUSD"), "BTC-USD")
        self.assertEqual(normalize_symbol("ETHUSD"), "ETH-USD")
        self.assertEqual(normalize_symbol("BTC/USD"), "BTC-USD")

    def test_index_cfds(self):
        self.assertEqual(normalize_symbol("SPX500"), "^GSPC")
        self.assertEqual(normalize_symbol("NAS100"), "^NDX")
        self.assertEqual(normalize_symbol("US30"), "^DJI")

    def test_detect_asset_type(self):
        self.assertEqual(detect_asset_type("EURUSD"), AssetType.FOREX)
        self.assertEqual(detect_asset_type("XAUUSD"), AssetType.COMMODITY)
        self.assertEqual(detect_asset_type("GOLD"), AssetType.COMMODITY)
        self.assertEqual(detect_asset_type("BTC-USD"), AssetType.CRYPTO)
        self.assertEqual(detect_asset_type("BBRI.JK"), AssetType.STOCK)
        self.assertEqual(detect_asset_type("NVDA"), AssetType.STOCK)

    def test_normalize_ticker_symbol_safe_path(self):
        self.assertEqual(normalize_ticker_symbol("EURUSD"), "EURUSD=X")
        self.assertEqual(normalize_ticker_symbol("XAUUSD+"), "GC=F")
        self.assertTrue(is_yahoo_safe("EURUSD=X"))
        self.assertTrue(is_yahoo_safe("GC=F"))
        err = NoMarketDataError("FOO", "BAR")
        self.assertIn("FOO", str(err))
        with self.assertRaises(ValueError):
            normalize_ticker_symbol("../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
