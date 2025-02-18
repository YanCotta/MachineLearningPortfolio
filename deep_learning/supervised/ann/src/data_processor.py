import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to handle all data preprocessing operations for the neural network.
    
    Attributes:
        scaler (StandardScaler): Scaler instance for feature normalization
        label_encoder (LabelEncoder): Encoder for categorical variables
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Testing labels
    """
    
    def __init__(self):
        """Initialize the DataProcessor with necessary preprocessing objects."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        try:
            logger.info(f"Loading data from {filepath}")
            data = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(data)} rows of data")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
            
    def preprocess_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for neural network training.
        
        Args:
            data (pd.DataFrame): Raw input data
            target_column (str): Name of the target variable column
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed features and target variables
        """
        logger.info("Starting data preprocessing")
        
        # Drop unnecessary columns
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        X = data.drop(columns=columns_to_drop + [target_column])
        
        # Handle categorical variables
        X = pd.get_dummies(X, columns=['Geography', 'Gender'])
        
        # Scale numerical features
        X = self.scaler.fit_transform(X)
        
        # Encode target variable
        y = self.label_encoder.fit_transform(data[target_column])
        
        logger.info("Data preprocessing completed successfully")
        return X, y
    
    def split_data(self, 
                X: np.ndarray, 
                y: np.ndarray, 
                test_size: float = 0.2, 
                validation_split: Optional[float] = 0.1,
                random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Split data into training, testing, and optionally validation sets.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            test_size (float): Proportion of dataset to include in the test split
            validation_split (Optional[float]): Proportion of training data to use for validation
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple[np.ndarray, ...]: Split datasets
        """
        logger.info("Splitting data into train and test sets")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if validation_split:
            logger.info("Creating validation split")
            val_size = validation_split / (1 - test_size)
            self.X_train, X_val, self.y_train, y_val = train_test_split(
                self.X_train, self.y_train, test_size=val_size, random_state=random_state
            )
            return self.X_train, X_val, self.X_test, self.y_train, y_val, self.y_test
            
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_preprocessed_data(self, 
                            filepath: str, 
                            target_column: str,
                            test_size: float = 0.2,
                            validation_split: Optional[float] = 0.1,
                            random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            filepath (str): Path to the data file
            target_column (str): Name of the target variable column
            test_size (float): Proportion of dataset to include in the test split
            validation_split (Optional[float]): Proportion of training data to use for validation
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple[np.ndarray, ...]: Preprocessed and split datasets
        """
        data = self.load_data(filepath)
        X, y = self.preprocess_data(data, target_column)
        return self.split_data(X, y, test_size, validation_split, random_state)
