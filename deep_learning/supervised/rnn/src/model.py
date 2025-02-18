import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
from typing import Tuple, Optional, Dict
import numpy as np
import os

logger = logging.getLogger(__name__)

class StockPredictor:
    """Advanced LSTM-based model for stock price prediction."""

    def __init__(
        self,
        sequence_length: int,
        n_features: int = 1,
        lstm_units: list = [100, 50, 50],
        dropout_rates: list = [0.3, 0.2, 0.2],
        bidirectional: bool = True,
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        save_path: str = 'models'
    ):
        """
        Initialize the model with configurable architecture.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            lstm_units: List of units for each LSTM layer
            dropout_rates: List of dropout rates for each layer
            bidirectional: Whether to use bidirectional LSTM
            use_batch_norm: Whether to use batch normalization
            learning_rate: Learning rate for Adam optimizer
            save_path: Directory to save model checkpoints
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build an advanced LSTM model architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        
        for i, (units, dropout_rate) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            # First layer needs input shape
            if i == 0:
                if self.bidirectional:
                    model.add(Bidirectional(
                        LSTM(units, return_sequences=(i < len(self.lstm_units)-1)),
                        input_shape=(self.sequence_length, self.n_features)
                    ))
                else:
                    model.add(LSTM(
                        units, 
                        return_sequences=(i < len(self.lstm_units)-1),
                        input_shape=(self.sequence_length, self.n_features)
                    ))
            else:
                if self.bidirectional:
                    model.add(Bidirectional(
                        LSTM(units, return_sequences=(i < len(self.lstm_units)-1))
                    ))
                else:
                    model.add(LSTM(
                        units, 
                        return_sequences=(i < len(self.lstm_units)-1)
                    ))
            
            if self.use_batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with Adam optimizer and MSE loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info(f"Model built with input shape: ({self.sequence_length}, {self.n_features})")
        return model

    def get_callbacks(self, patience: int = 10) -> list:
        """
        Get training callbacks for early stopping and model checkpointing.
        
        Args:
            patience: Number of epochs to wait for improvement
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.save_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                verbose=1,
                min_lr=1e-6
            )
        ]
        return callbacks

    def train(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 10) -> tf.keras.callbacks.History:
        """
        Train the model with early stopping and learning rate scheduling.
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_data: Optional tuple of validation features and targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
        # Get callbacks
        callbacks = self.get_callbacks(patience=patience)
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """Load a saved model from disk."""
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """Get model architecture summary as string."""
        from io import StringIO
        summary = StringIO()
        self.model.summary(print_fn=lambda x: summary.write(x + '\n'))
        return summary.getvalue()