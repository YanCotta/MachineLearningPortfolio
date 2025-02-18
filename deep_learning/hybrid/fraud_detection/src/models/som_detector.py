import numpy as np
from minisom import MiniSom
import logging

logger = logging.getLogger(__name__)

class SOMFraudDetector:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        """Initialize the SOM for fraud detection."""
        self.som = MiniSom(
            x=x, y=y,
            input_len=input_len,
            sigma=sigma,
            learning_rate=learning_rate
        )
        self.x = x
        self.y = y
        
    def train(self, X, num_iterations=100):
        """Train the SOM on the input data."""
        try:
            # Initialize weights randomly
            self.som.random_weights_init(X)
            
            # Train the SOM
            self.som.train_random(
                data=X,
                num_iteration=num_iterations
            )
            logger.info(f"SOM training completed with {num_iterations} iterations")
            
        except Exception as e:
            logger.error(f"Error during SOM training: {str(e)}")
            raise
    
    def detect_anomalies(self, X, threshold=0.9):
        """Detect anomalies using trained SOM.
        
        Args:
            X: Input data to analyze
            threshold: Percentile threshold for anomaly detection
            
        Returns:
            indices of detected anomalies
        """
        try:
            # Calculate anomaly scores
            anomaly_scores = self.get_anomaly_scores(X)
            
            # Get threshold value based on percentile
            threshold_value = np.percentile(anomaly_scores, threshold * 100)
            
            # Find anomalies above threshold
            anomaly_indices = np.where(anomaly_scores > threshold_value)[0]
            
            logger.info(f"Detected {len(anomaly_indices)} anomalies using threshold {threshold}")
            return anomaly_indices
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    def get_anomaly_scores(self, X):
        """Calculate anomaly scores for each data point."""
        return np.array([self.som.quantization_error(x) for x in X])
    
    def get_feature_map(self):
        """Get the SOM feature map for visualization."""
        return self.som.distance_map()
    
    def get_winning_nodes(self, X):
        """Get winning nodes for each input vector."""
        winners = np.array([self.som.winner(x) for x in X])
        return winners