import unittest
from tradingagents.graph.setup import DEBATE_PATH_MAP, RISK_ANALYSIS_PATH_MAP


class TestGraphPathMaps(unittest.TestCase):
    def test_debate_path_map_completeness(self):
        expected_debate_keys = {"Bull Researcher", "Bear Researcher", "Research Manager"}
        self.assertEqual(set(DEBATE_PATH_MAP.keys()), expected_debate_keys)
        self.assertEqual(set(DEBATE_PATH_MAP.values()), expected_debate_keys)

    def test_risk_path_map_completeness(self):
        expected_risk_keys = {
            "Aggressive Analyst",
            "Conservative Analyst",
            "Neutral Analyst",
            "Portfolio Manager",
        }
        self.assertEqual(set(RISK_ANALYSIS_PATH_MAP.keys()), expected_risk_keys)
        self.assertEqual(set(RISK_ANALYSIS_PATH_MAP.values()), expected_risk_keys)


if __name__ == "__main__":
    unittest.main()
