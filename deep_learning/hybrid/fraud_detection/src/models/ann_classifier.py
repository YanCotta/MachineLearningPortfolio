import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import logging

logger = logging.getLogger(__name__)

class ANNClassifier:
    def __init__(self, input_dim, hidden_layers=[6, 6], dropout_rate=0.3):
        """Initialize ANN classifier.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons for each hidden layer
            dropout_rate: Dropout rate for regularization
        """
        self.model = self._build_model(input_dim, hidden_layers, dropout_rate)
        
    def _build_model(self, input_dim, hidden_layers, dropout_rate):
        """Build the neural network architecture."""
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the neural network."""
        try:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            logger.info("ANN training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during ANN training: {str(e)}")
            raise
    
    def predict(self, X, threshold=0.5):
        """Make predictions using the trained model."""
        try:
            predictions = self.model.predict(X)
            return (predictions >= threshold).astype(int).ravel()
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_feature_importance(self, X):
        """Calculate feature importance using gradient-based approach."""
        try:
            # Create gradient model
            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=self.model.outputs
            )
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                inputs = tf.cast(X, tf.float32)
                tape.watch(inputs)
                predictions = grad_model(inputs)
            
            gradients = tape.gradient(predictions, inputs)
            importance = np.abs(gradients.numpy()).mean(axis=0)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise