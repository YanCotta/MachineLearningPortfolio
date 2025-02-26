import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path):
        """Initialize DataLoader with path to dataset."""
        self.data_path = Path(data_path)
        self.feature_names = None
        self.scaler = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
    
    def load_and_preprocess(self):
        """Load and preprocess the credit card application data."""
        try:
            # Load dataset
            dataset = pd.read_csv(self.data_path)
            
            if dataset.empty:
                raise ValueError("Dataset is empty")
                
            logger.info(f"Loaded dataset with shape: {dataset.shape}")
            
            # Check for missing values
            if dataset.isnull().any().any():
                logger.warning("Dataset contains missing values. Handling them...")
                dataset = dataset.fillna(dataset.mean())
            
            # Extract features and store feature names
            X = dataset.iloc[:, :-1].values  # All columns except last
            y = dataset.iloc[:, -1].values   # Last column is the class
            self.feature_names = dataset.columns[:-1].tolist()
            
            # Validate target variable
            if not set(np.unique(y)).issubset({0, 1}):
                raise ValueError("Target variable contains invalid values")
            
            # Store original data for reference
            X_original = X.copy()
            
            # Handle any infinite values
            X[~np.isfinite(X)] = 0
            logger.info("Cleaned infinite values in dataset")
            
            # Scale features using MinMaxScaler for SOM
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = self.scaler.fit_transform(X)
            
            # Validate scaled data
            if not np.all(np.isfinite(X_scaled)):
                raise ValueError("Scaling produced invalid values")
            
            logger.info("Data preprocessing completed successfully")
            return X_scaled, y, X_original
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def get_train_test_split(self, X, y, test_size=0.2, stratify=True):
        """Split data into training and test sets."""
        if len(X) != len(y):
            raise ValueError("Features and target arrays must have the same length")
            
        stratify_param = y if stratify else None
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_param
        )
    
    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call load_and_preprocess first.")
            
        if not np.all(np.isfinite(X_scaled)):
            raise ValueError("Input contains invalid values")
            
        return self.scaler.inverse_transform(X_scaled)