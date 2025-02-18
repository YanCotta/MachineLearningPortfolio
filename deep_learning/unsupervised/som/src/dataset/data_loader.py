import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class CreditCardDataLoader:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")
        self.filepath = filepath
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """Load and preprocess the credit card application dataset.
        
        Returns:
            tuple: (X_scaled, y, X_original) where:
                - X_scaled: Scaled feature matrix
                - y: Target labels
                - X_original: Original unscaled features
        
        Raises:
            ValueError: If the dataset format is invalid
        """
        try:
            dataset = pd.read_csv(self.filepath)
            
            if len(dataset.columns) < 2:
                raise ValueError("Dataset must contain features and target column")
                
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            
            # Validate data types
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("Target values must be numeric")
            
            # Check for missing values
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("Dataset contains missing values")
            
            # Feature scaling
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y, X
            
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {str(e)}")
    
    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            numpy.ndarray: Original scale features
            
        Raises:
            ValueError: If scaler has not been fitted
        """
        if not hasattr(self.scaler, 'scale_'):
            raise ValueError("Scaler has not been fitted. Call load_data() first")
        return self.scaler.inverse_transform(X_scaled)