import unittest
# ...existing code...
from src.market_with_apriori import MarketAnalyzer
from src.market_with_eclat import EclatAnalyzer

class TestAssociationRules(unittest.TestCase):
    """Tests for Apriori and ECLAT analyzers."""

    def test_apriori_init(self):
        analyzer = MarketAnalyzer()
        self.assertIsNotNone(analyzer)

    def test_eclat_init(self):
        analyzer = EclatAnalyzer()
        self.assertIsNotNone(analyzer)
