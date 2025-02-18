import unittest
import numpy as np
import os
from src.dataset.data_loader import CreditCardDataLoader
from src.model.som_model import CreditCardSOM

class TestCreditCardFraudDetection(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_data = np.random.rand(100, 15)
        self.test_labels = np.random.randint(0, 2, 100)
        
    def test_som_initialization(self):
        """Test SOM model initialization."""
        som = CreditCardSOM(x=10, y=10, input_len=15)
        self.assertFalse(som.is_trained)
        self.assertEqual(som.som._weights.shape, (10, 10, 15))
        
    def test_som_training(self):
        """Test SOM model training."""
        som = CreditCardSOM(x=10, y=10, input_len=15)
        som.train(self.test_data)
        self.assertTrue(som.is_trained)
        
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        som = CreditCardSOM(x=10, y=10, input_len=15)
        som.train(self.test_data)
        anomalies = som.find_anomalies(self.test_data)
        self.assertEqual(len(anomalies), len(self.test_data))
        self.assertTrue(isinstance(anomalies, np.ndarray))
        
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        som = CreditCardSOM(x=10, y=10, input_len=15)
        with self.assertRaises(ValueError):
            som.train(np.random.rand(100, 14))  # Wrong number of features
            
    def test_data_loader(self):
        """Test data loader with sample data."""
        test_data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Credit_Card_Applications.csv')
        if os.path.exists(test_data_path):
            loader = CreditCardDataLoader(test_data_path)
            X_scaled, y, X = loader.load_data()
            self.assertEqual(X_scaled.shape[1], 15)
            self.assertTrue(np.all(X_scaled >= 0) and np.all(X_scaled <= 1))

if __name__ == '__main__':
    unittest.main()