import unittest
from src.natural_language_processing import main

class TestNLP(unittest.TestCase):
    def test_main_runs(self):
        # Ensure main function executes without errors
        try:
            main()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Main function failed due to {e}")

if __name__ == '__main__':
    unittest.main()
