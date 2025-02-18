import unittest
import numpy as np
import tempfile
from pathlib import Path
from src.visualization.visualizer import FraudVisualization
from src.models.som_detector import SOMFraudDetector
from src.models.ann_classifier import ANNClassifier

class TestFraudVisualization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = FraudVisualization(self.temp_dir)
        
        # Set up test data
        self.input_dim = 10
        self.n_samples = 100
        self.X = np.random.rand(self.n_samples, self.input_dim)
        self.y = np.random.randint(0, 2, self.n_samples)
        self.feature_names = [f'Feature_{i}' for i in range(self.input_dim)]
        
        # Set up models
        self.som = SOMFraudDetector(x=5, y=5, input_len=self.input_dim)
        self.som.train(self.X)
        
        self.ann = ANNClassifier(input_dim=self.input_dim)
        self.ann.train(self.X, self.y, epochs=2)  # Few epochs for testing
    
    def test_output_directory_creation(self):
        # Test with non-existent directory
        new_dir = Path(self.temp_dir) / 'new_output'
        visualizer = FraudVisualization(str(new_dir))
        self.assertTrue(new_dir.exists())
    
    def test_som_heatmap(self):
        # Test untrained SOM validation
        untrained_som = SOMFraudDetector(x=5, y=5, input_len=self.input_dim)
        with self.assertRaises(RuntimeError):
            self.visualizer.plot_som_heatmap(untrained_som, self.X, self.y)
        
        # Test successful plotting
        self.visualizer.plot_som_heatmap(self.som, self.X, self.y)
        heatmap_file = Path(self.temp_dir) / 'som_heatmap.png'
        self.assertTrue(heatmap_file.exists())
    
    def test_fraud_distribution(self):
        predictions = self.ann.predict(self.X)
        
        # Test length mismatch
        with self.assertRaises(ValueError):
            self.visualizer.plot_fraud_distribution(
                predictions[:50],  # Wrong length
                self.X,
                self.feature_names
            )
        
        # Test successful plotting
        self.visualizer.plot_fraud_distribution(
            predictions,
            self.X,
            self.feature_names
        )
        dist_file = Path(self.temp_dir) / 'fraud_distribution.png'
        self.assertTrue(dist_file.exists())
    
    def test_feature_importance(self):
        # Test untrained model validation
        untrained_ann = ANNClassifier(input_dim=self.input_dim)
        with self.assertRaises(RuntimeError):
            self.visualizer.plot_feature_importance(
                untrained_ann,
                self.feature_names,
                self.X
            )
        
        # Test feature names length mismatch
        with self.assertRaises(ValueError):
            self.visualizer.plot_feature_importance(
                self.ann,
                self.feature_names[:-1],  # Wrong length
                self.X
            )
        
        # Test successful plotting
        self.visualizer.plot_feature_importance(
            self.ann,
            self.feature_names,
            self.X
        )
        importance_file = Path(self.temp_dir) / 'feature_importance.png'
        self.assertTrue(importance_file.exists())
    
    def test_model_metrics(self):
        predictions = self.ann.predict(self.X)
        
        # Test length mismatch
        with self.assertRaises(ValueError):
            self.visualizer.plot_model_metrics(
                self.y,
                predictions[:50]  # Wrong length
            )
        
        # Test successful plotting
        self.visualizer.plot_model_metrics(self.y, predictions)
        metrics_file = Path(self.temp_dir) / 'confusion_matrix.png'
        self.assertTrue(metrics_file.exists())
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()