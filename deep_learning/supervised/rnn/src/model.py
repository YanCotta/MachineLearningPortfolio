import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, sequence_length: int, save_path: str = 'models'):
        """
        Initialize the stock predictor model
        
        Args:
            sequence_length: Number of time steps to look back
            save_path: Directory to save model checkpoints
        """
        self.sequence_length = sequence_length
        self.save_path = save_path
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build the LSTM model architecture
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info(f"Model built with input shape: ({self.sequence_length}, 1)")
        return model
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             epochs: int = 100,
             batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the model on the provided data
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_data: Optional tuple of validation features and targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.save_path, 'best_model.h5'),
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True
            ),
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Reshape input data
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        if validation_data:
            X_val, y_val = validation_data
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            validation_data = (X_val_reshaped, y_val)
        
        # Train the model
        history = self.model.fit(
            X_train_reshaped,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        predictions = self.model.predict(X_reshaped)
        return predictions
    
    def load_weights(self, weights_path: str):
        """
        Load model weights from a file
        
        Args:
            weights_path: Path to the weights file
        """
        self.model.load_weights(weights_path)
        logger.info(f"Loaded model weights from {weights_path}")
    
    def save_weights(self, weights_path: str):
        """
        Save model weights to a file
        
        Args:
            weights_path: Path to save the weights
        """
        self.model.save_weights(weights_path)
        logger.info(f"Saved model weights to {weights_path}")