import unittest
import numpy as np
from src.data_processing.data_loader import DataLoader
from pathlib import Path

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.parent / "data" / "Credit_Card_Applications.csv"
        self.data_loader = DataLoader(self.data_path)
    
    def test_load_and_preprocess(self):
        X_scaled, y, X_original = self.data_loader.load_and_preprocess()
        
        # Check shapes match
        self.assertEqual(X_scaled.shape, X_original.shape)
        self.assertEqual(len(y), len(X_scaled))
        
        # Check scaling is correct
        self.assertTrue(np.all(X_scaled >= 0))
        self.assertTrue(np.all(X_scaled <= 1))
        
        # Check no missing values
        self.assertTrue(np.all(np.isfinite(X_scaled)))
        
        # Check target is binary
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))
    
    def test_train_test_split(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split(X, y)
        
        # Check sizes
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        
        # Check proportions
        self.assertAlmostEqual(len(X_test) / len(X), 0.2, places=1)
    
    def test_inverse_transform(self):
        X = np.random.rand(100, 10)
        self.data_loader.scaler = None
        
        # Should raise error if scaler not fitted
        with self.assertRaises(ValueError):
            self.data_loader.inverse_transform(X)
            
if __name__ == '__main__':
    unittest.main()