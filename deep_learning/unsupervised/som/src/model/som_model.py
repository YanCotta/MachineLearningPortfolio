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
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not all(isinstance(param, (int, float)) for param in [x, y, input_len, sigma, learning_rate]):
            raise ValueError("All parameters must be numeric")
        if not all(param > 0 for param in [x, y, input_len, sigma, learning_rate]):
            raise ValueError("All parameters must be positive")
            
        self.som = MiniSom(x=x, y=y, input_len=input_len, 
                          sigma=sigma, learning_rate=learning_rate)
        self.is_trained = False
        
    def train(self, X, num_iterations=100):
        """Train the SOM on the input data.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            num_iterations (int): Number of training iterations
            
        Raises:
            ValueError: If input data is invalid or incompatible
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        
        if X.shape[1] != self.som._weights.shape[2]:
            raise ValueError(f"Input data must have {self.som._weights.shape[2]} features")
            
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer")
            
        self.som.random_weights_init(X)
        self.som.train_random(data=X, num_iteration=num_iterations)
        self.is_trained = True
        
    def get_anomaly_scores(self, X):
        """Get anomaly scores for input data points.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            list: List of anomaly scores for each input point
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input data is invalid
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting anomaly scores")
            
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")
            
        if X.shape[1] != self.som._weights.shape[2]:
            raise ValueError(f"Input data must have {self.som._weights.shape[2]} features")
            
        return [self.som.distance_map()[self.som.winner(x)] for x in X]
    
    def find_anomalies(self, X, threshold=0.9):
        """Find anomalies in the dataset based on quantile threshold.
        
        Args:
            X (np.ndarray): Input data
            threshold (float): Quantile threshold for anomaly detection
            
        Returns:
            np.ndarray: Boolean mask indicating anomalies
            
        Raises:
            ValueError: If threshold is invalid
        """
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
            
        scores = self.get_anomaly_scores(X)
        threshold_value = np.quantile(scores, threshold)
        return np.array(scores) > threshold_value
    
    def get_feature_map(self):
        """Get the SOM distance map for visualization.
        
        Returns:
            np.ndarray: Distance map matrix
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature map")
        return self.som.distance_map()
    
    def get_winning_nodes(self, X):
        """Get winning nodes for each input data point.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            list: Winning nodes for each input point
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting winning nodes")
        return [self.som.winner(x) for x in X]