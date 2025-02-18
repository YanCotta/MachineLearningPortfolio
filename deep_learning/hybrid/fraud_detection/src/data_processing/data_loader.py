import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path):
        """Initialize DataLoader with path to dataset."""
        self.data_path = data_path
        self.feature_names = None
        self.scaler = None
    
    def load_and_preprocess(self):
        """Load and preprocess the credit card application data."""
        try:
            # Load dataset
            dataset = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {dataset.shape}")
            
            # Extract features and store feature names
            X = dataset.iloc[:, :-1].values  # All columns except last
            y = dataset.iloc[:, -1].values   # Last column is the class
            self.feature_names = dataset.columns[:-1].tolist()
            
            # Store original data for reference
            X_original = X.copy()
            
            # Scale features using MinMaxScaler for SOM
            # (SOM works better with data scaled between 0 and 1)
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = self.scaler.fit_transform(X)
            
            logger.info("Data preprocessing completed successfully")
            return X_scaled, y, X_original
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def get_train_test_split(self, X, y, test_size=0.2):
        """Split data into training and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call load_and_preprocess first.")
        return self.scaler.inverse_transform(X_scaled)