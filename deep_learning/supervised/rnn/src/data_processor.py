import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data preprocessing for stock price prediction."""
    
    def __init__(self, scaler: Optional[MinMaxScaler] = None):
        """
        Initialize the data processor.
        
        Args:
            scaler: Optional pre-fitted scaler. If None, creates new one.
        """
        self.scaler = scaler or MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate stock price data.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with processed stock data
        """
        try:
            df = pd.read_csv(filepath)
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Validate columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Handle missing values
            if df.isnull().any().any():
                logger.warning("Missing values detected. Filling with forward fill method.")
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)  # Backup for any remaining NaNs
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_sequences(self, 
                        data: np.ndarray, 
                        sequence_length: int,
                        target_column: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data array
            sequence_length: Length of input sequences
            target_column: Index of target column (default: -1 for last column)
            
        Returns:
            Tuple of (X, y) where X contains sequences and y contains targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, target_column])
            
        return np.array(X), np.array(y)
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    sequence_length: int,
                    train_split: float = 0.8,
                    features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            train_split: Proportion of data for training
            features: List of feature columns to use (default: ['Close'])
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        features = features or ['Close']
        data = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.prepare_sequences(scaled_data, sequence_length)
        
        # Split into train and test
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Data prepared with shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Transform scaled values back to original scale."""
        # Reshape if needed
        reshaped_data = data.reshape(-1, 1) if len(data.shape) == 1 else data
        return self.scaler.inverse_transform(reshaped_data)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis indicators to the dataset.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Moving averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Fill NaN values created by indicators
        df.fillna(method='bfill', inplace=True)
        
        return df