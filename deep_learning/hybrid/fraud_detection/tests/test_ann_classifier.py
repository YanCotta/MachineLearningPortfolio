import unittest
import numpy as np
import tensorflow as tf
from src.models.ann_classifier import ANNClassifier

class TestANNClassifier(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.ann = ANNClassifier(
            input_dim=self.input_dim,
            hidden_layers=[8, 4],
            dropout_rate=0.3
        )
        self.X = np.random.rand(100, self.input_dim)
        self.y = np.random.randint(0, 2, 100)
    
    def test_init_validation(self):
        # Test invalid input dimension
        with self.assertRaises(ValueError):
            ANNClassifier(input_dim=0)
        
        # Test invalid hidden layers
        with self.assertRaises(ValueError):
            ANNClassifier(input_dim=10, hidden_layers=[])
            
        # Test invalid dropout rate
        with self.assertRaises(ValueError):
            ANNClassifier(input_dim=10, dropout_rate=-0.1)
        with self.assertRaises(ValueError):
            ANNClassifier(input_dim=10, dropout_rate=1.5)
    
    def test_model_architecture(self):
        # Check input shape
        self.assertEqual(
            self.ann.model.input_shape,
            (None, self.input_dim)
        )
        
        # Check output shape
        self.assertEqual(
            self.ann.model.output_shape,
            (None, 1)
        )
        
        # Verify layer types
        layers = self.ann.model.layers
        self.assertIsInstance(layers[0], tf.keras.layers.Dense)
        self.assertIsInstance(layers[1], tf.keras.layers.BatchNormalization)
        self.assertIsInstance(layers[2], tf.keras.layers.Dropout)
    
    def test_train_validation(self):
        # Test wrong input dimension
        wrong_X = np.random.rand(100, self.input_dim + 1)
        with self.assertRaises(ValueError):
            self.ann.train(wrong_X, self.y)
            
        # Test non-binary target
        wrong_y = np.random.randint(0, 3, 100)
        with self.assertRaises(ValueError):
            self.ann.train(self.X, wrong_y)
            
        # Test mismatched lengths
        wrong_y = np.random.randint(0, 2, 50)
        with self.assertRaises(ValueError):
            self.ann.train(self.X, wrong_y)
    
    def test_training_workflow(self):
        # Test training completion
        history = self.ann.train(
            self.X, self.y,
            epochs=2,  # Small number for testing
            batch_size=32
        )
        self.assertTrue(self.ann._trained)
        
        # Verify history object
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
        
        # Test predictions
        preds = self.ann.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        
        # Test feature importance
        importance = self.ann.get_feature_importance(self.X)
        self.assertEqual(len(importance), self.input_dim)
        self.assertTrue(np.all(np.isfinite(importance)))
    
    def test_untrained_validation(self):
        # All operations should fail if model is not trained
        untrained_ann = ANNClassifier(input_dim=self.input_dim)
        
        with self.assertRaises(RuntimeError):
            untrained_ann.predict(self.X)
            
        with self.assertRaises(RuntimeError):
            untrained_ann.get_feature_importance(self.X)
            
if __name__ == '__main__':
    unittest.main()