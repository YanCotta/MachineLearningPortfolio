import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import logging

logger = logging.getLogger(__name__)

class ANNClassifier:
    def __init__(self, input_dim, hidden_layers=[6, 6], dropout_rate=0.3):
        """Initialize ANN classifier."""
        # Validate input parameters
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive")
        if not hidden_layers:
            raise ValueError("Must specify at least one hidden layer")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
            
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        self._trained = False
        
    def _build_model(self):
        """Build the neural network architecture."""
        try:
            model = Sequential()
            
            # Input layer
            model.add(Dense(
                self.hidden_layers[0],
                input_dim=self.input_dim,
                activation='relu',
                kernel_initializer='he_uniform'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
            
            # Hidden layers
            for units in self.hidden_layers[1:]:
                if units <= 0:
                    raise ValueError("Number of units must be positive")
                model.add(Dense(units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout_rate))
            
            # Output layer
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model with optimization configs
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the neural network."""
        try:
            # Validate input data
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError("Inputs must be numpy arrays")
            if X.shape[1] != self.input_dim:
                raise ValueError(f"Expected input dimension {self.input_dim}, got {X.shape[1]}")
            if not set(np.unique(y)).issubset({0, 1}):
                raise ValueError("Target variable must be binary")
            if len(X) != len(y):
                raise ValueError("Features and target must have same length")
                
            # Validate training parameters
            if not 0 < validation_split < 1:
                raise ValueError("Validation split must be between 0 and 1")
            if epochs <= 0 or batch_size <= 0:
                raise ValueError("Epochs and batch size must be positive")
                
            # Configure callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            # Train model
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            self._trained = True
            logger.info("ANN training completed successfully")
            
            # Log training results
            val_accuracy = max(history.history['val_accuracy'])
            logger.info(f"Best validation accuracy: {val_accuracy:.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during ANN training: {str(e)}")
            raise
    
    def predict(self, X, threshold=0.5):
        """Make predictions using the trained model."""
        if not self._trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        try:
            # Validate input
            if X.shape[1] != self.input_dim:
                raise ValueError("Input dimension mismatch")
            
            predictions = self.model.predict(X, verbose=0)
            return (predictions >= threshold).astype(int).ravel()
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_feature_importance(self, X):
        """Calculate feature importance using gradient-based approach."""
        if not self._trained:
            raise RuntimeError("Model must be trained before calculating importance")
            
        try:
            # Validate input
            if X.shape[1] != self.input_dim:
                raise ValueError("Input dimension mismatch")
            
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
            
            if not np.all(np.isfinite(importance)):
                raise ValueError("Invalid importance scores computed")
                
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise