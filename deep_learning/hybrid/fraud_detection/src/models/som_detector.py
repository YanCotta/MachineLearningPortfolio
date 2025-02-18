import numpy as np
from minisom import MiniSom
import logging

logger = logging.getLogger(__name__)

class SOMFraudDetector:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        """Initialize the SOM for fraud detection."""
        # Validate input parameters
        if x <= 0 or y <= 0:
            raise ValueError("Grid dimensions must be positive")
        if input_len <= 0:
            raise ValueError("Input dimension must be positive")
        if sigma <= 0 or learning_rate <= 0:
            raise ValueError("Sigma and learning rate must be positive")
            
        self.som = MiniSom(
            x=x, y=y,
            input_len=input_len,
            sigma=sigma,
            learning_rate=learning_rate
        )
        self.x = x
        self.y = y
        self._trained = False
        
    def train(self, X, num_iterations=100):
        """Train the SOM on the input data."""
        try:
            # Validate input data
            if not isinstance(X, np.ndarray):
                raise TypeError("Input data must be a numpy array")
            if X.shape[1] != self.som.input_len:
                raise ValueError(f"Expected input dimension {self.som.input_len}, got {X.shape[1]}")
            if not np.all(np.isfinite(X)):
                raise ValueError("Input contains invalid values")
            
            # Initialize weights randomly
            self.som.random_weights_init(X)
            
            # Train the SOM
            self.som.train_random(
                data=X,
                num_iteration=num_iterations,
                verbose=False
            )
            
            self._trained = True
            logger.info(f"SOM training completed with {num_iterations} iterations")
            
        except Exception as e:
            logger.error(f"Error during SOM training: {str(e)}")
            raise
    
    def detect_anomalies(self, X, threshold=0.9):
        """Detect anomalies using trained SOM."""
        if not self._trained:
            raise RuntimeError("SOM must be trained before detecting anomalies")
            
        try:
            # Validate input
            if X.shape[1] != self.som.input_len:
                raise ValueError("Input dimension mismatch")
                
            # Calculate anomaly scores
            anomaly_scores = self.get_anomaly_scores(X)
            
            # Validate scores
            if not np.all(np.isfinite(anomaly_scores)):
                raise ValueError("Invalid anomaly scores computed")
            
            # Get threshold value based on percentile
            threshold_value = np.percentile(anomaly_scores, threshold * 100)
            
            # Find anomalies above threshold
            anomaly_indices = np.where(anomaly_scores > threshold_value)[0]
            
            # Warn if no anomalies found
            if len(anomaly_indices) == 0:
                logger.warning("No anomalies detected with current threshold")
            else:
                logger.info(f"Detected {len(anomaly_indices)} anomalies using threshold {threshold}")
                
            return anomaly_indices
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    def get_anomaly_scores(self, X):
        """Calculate anomaly scores for each data point."""
        if not self._trained:
            raise RuntimeError("SOM must be trained before calculating scores")
            
        try:
            scores = np.array([self.som.quantization_error(x) for x in X])
            if not np.all(np.isfinite(scores)):
                raise ValueError("Invalid scores computed")
            return scores
        except Exception as e:
            logger.error(f"Error calculating anomaly scores: {str(e)}")
            raise
    
    def get_feature_map(self):
        """Get the SOM feature map for visualization."""
        if not self._trained:
            raise RuntimeError("SOM must be trained before getting feature map")
        return self.som.distance_map()
    
    def get_winning_nodes(self, X):
        """Get winning nodes for each input vector."""
        if not self._trained:
            raise RuntimeError("SOM must be trained before getting winning nodes")
            
        try:
            winners = np.array([self.som.winner(x) for x in X])
            return winners
        except Exception as e:
            logger.error(f"Error getting winning nodes: {str(e)}")
            raise