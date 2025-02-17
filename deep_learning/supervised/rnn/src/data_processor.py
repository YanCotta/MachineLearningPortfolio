import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, train_path: str, test_path: str):
        """
        Initialize the data processor with paths to training and test data
        
        Args:
            train_path: Path to training dataset
            test_path: Path to test dataset
        """
        self.train_path = train_path
        self.test_path = test_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # Number of time steps to look back
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the data for training and testing
        
        Returns:
            Tuple containing X_train, y_train, X_test, y_test
        """
        try:
            # Load the datasets
            train_data = pd.read_csv(self.train_path)
            test_data = pd.read_csv(self.test_path)
            
            logger.info(f"Loaded training data shape: {train_data.shape}")
            logger.info(f"Loaded test data shape: {test_data.shape}")
            
            # Extract and transform the 'Close' price
            training_set = train_data['Close'].values.reshape(-1, 1)
            test_set = test_data['Close'].values.reshape(-1, 1)
            
            # Scale the data
            training_set_scaled = self.scaler.fit_transform(training_set)
            test_set_scaled = self.scaler.transform(test_set)
            
            # Prepare training data
            X_train, y_train = self._create_sequences(training_set_scaled)
            
            # Prepare test data
            X_test, y_test = self._create_sequences(test_set_scaled)
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of X and y arrays
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale
        
        Args:
            scaled_data: Scaled predictions
            
        Returns:
            Data in original scale
        """
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1))