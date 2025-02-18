from minisom import MiniSom
import numpy as np

class CreditCardSOM:
    def __init__(self, x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5):
        """Initialize the SOM model for credit card fraud detection.
        
        Args:
            x (int): Width of the SOM grid
            y (int): Height of the SOM grid
            input_len (int): Number of features in input data
            sigma (float): The radius of the neighborhood function
            learning_rate (float): The learning rate for weight updates
        """
        self.som = MiniSom(x=x, y=y, input_len=input_len, 
                        sigma=sigma, learning_rate=learning_rate)
        
    def train(self, X, num_iterations=100):
        """Train the SOM on the input data.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            num_iterations (int): Number of training iterations
        """
        self.som.random_weights_init(X)
        self.som.train_random(data=X, num_iteration=num_iterations)
        
    def get_anomaly_scores(self, X):
        """Get anomaly scores for input data points.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            list: List of anomaly scores for each input point
        """
        return [self.som.distance_map()[self.som.winner(x)] for x in X]
    
    def find_anomalies(self, X, threshold=0.9):
        """Find anomalies in the dataset based on quantile threshold.
        
        Args:
            X (np.ndarray): Input data
            threshold (float): Quantile threshold for anomaly detection
            
        Returns:
            np.ndarray: Boolean mask indicating anomalies
        """
        scores = self.get_anomaly_scores(X)
        threshold_value = np.quantile(scores, threshold)
        return np.array(scores) > threshold_value
    
    def get_feature_map(self):
        """Get the SOM distance map for visualization."""
        return self.som.distance_map()
    
    def get_winning_nodes(self, X):
        """Get winning nodes for each input data point."""
        return [self.som.winner(x) for x in X]