import unittest
import numpy as np
from src.models.som_detector import SOMFraudDetector

class TestSOMDetector(unittest.TestCase):
    def setUp(self):
        self.input_len = 10
        self.som = SOMFraudDetector(x=5, y=5, input_len=self.input_len)
        self.X = np.random.rand(100, self.input_len)
    
    def test_init_validation(self):
        # Test invalid dimensions
        with self.assertRaises(ValueError):
            SOMFraudDetector(x=0, y=5, input_len=10)
        with self.assertRaises(ValueError):
            SOMFraudDetector(x=5, y=-1, input_len=10)
            
        # Test invalid input length
        with self.assertRaises(ValueError):
            SOMFraudDetector(x=5, y=5, input_len=0)
            
        # Test invalid learning parameters
        with self.assertRaises(ValueError):
            SOMFraudDetector(x=5, y=5, input_len=10, sigma=0)
        with self.assertRaises(ValueError):
            SOMFraudDetector(x=5, y=5, input_len=10, learning_rate=-1)
    
    def test_train_validation(self):
        # Test wrong input dimension
        wrong_X = np.random.rand(100, self.input_len + 1)
        with self.assertRaises(ValueError):
            self.som.train(wrong_X)
            
        # Test invalid values
        invalid_X = np.full((100, self.input_len), np.inf)
        with self.assertRaises(ValueError):
            self.som.train(invalid_X)
    
    def test_training_workflow(self):
        # Test training completion
        self.som.train(self.X)
        self.assertTrue(self.som._trained)
        
        # Get anomalies
        anomalies = self.som.detect_anomalies(self.X)
        self.assertIsInstance(anomalies, np.ndarray)
        self.assertTrue(len(anomalies) <= len(self.X))
        
        # Get feature map
        fmap = self.som.get_feature_map()
        self.assertEqual(fmap.shape, (self.som.x, self.som.y))
        
        # Get winning nodes
        winners = self.som.get_winning_nodes(self.X)
        self.assertEqual(len(winners), len(self.X))
    
    def test_untrained_validation(self):
        # All operations should fail if SOM is not trained
        untrained_som = SOMFraudDetector(x=5, y=5, input_len=self.input_len)
        
        with self.assertRaises(RuntimeError):
            untrained_som.detect_anomalies(self.X)
        
        with self.assertRaises(RuntimeError):
            untrained_som.get_feature_map()
            
        with self.assertRaises(RuntimeError):
            untrained_som.get_winning_nodes(self.X)
            
if __name__ == '__main__':
    unittest.main()