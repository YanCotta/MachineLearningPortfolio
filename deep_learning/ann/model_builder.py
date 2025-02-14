"""
Neural Network Model Builder for Bank Customer Churn Prediction
This module implements a deep neural network using TensorFlow/Keras for predicting customer churn.
The architecture includes multiple dense layers with batch normalization and dropout for regularization.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class BankChurnModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a more complex neural network"""
        model = Sequential([
            Dense(32, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with early stopping"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
