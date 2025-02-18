import unittest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test suites
from tests.test_data_loader import TestDataLoader
from tests.test_som_detector import TestSOMDetector
from tests.test_ann_classifier import TestANNClassifier
from tests.test_visualizer import TestFraudVisualization

def create_test_suite():
    """Create and return test suite with all tests."""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataLoader))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSOMDetector))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestANNClassifier))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFraudVisualization))
    
    return suite

if __name__ == '__main__':
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    runner.run(test_suite)