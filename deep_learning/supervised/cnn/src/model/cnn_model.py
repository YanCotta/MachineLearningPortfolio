"""
CNN model architecture and configuration.

This module defines the CNN architecture for binary image classification,
implementing a modern and efficient network design with batch normalization
and dropout for regularization.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class CNNModel:
    """Implements a CNN architecture for binary image classification."""

    def __init__(self, img_size: Tuple[int, int] = (64, 64)):
        """
        Initialize the CNN model.

        Args:
            img_size: Tuple of (height, width) for input images
        """
        self.img_size = img_size
        self.model = None

    def build(self) -> None:
        """
        Build the CNN architecture with modern best practices.
        
        The architecture includes:
        - Multiple convolutional layers with increasing filters
        - Batch normalization for training stability
        - MaxPooling for spatial dimension reduction
        - Dropout for regularization
        - Dense layers for final classification
        """
        self.model = models.Sequential([
            # Input Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=[*self.img_size, 3]),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Dropout(0.25),
            
            # Middle Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Dropout(0.25),
            
            # Deep Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Dropout(0.25),
            
            # Dense Classification Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile with modern optimization settings
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        logger.info("Model built successfully")
        self.model.summary()

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks for improved training process.

        Returns:
            List of Keras callbacks for training
        """
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path where to save the model
        """
        if self.model is None:
            raise ValueError("Model hasn't been built yet")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.

        Returns:
            String containing model summary
        """
        if self.model is None:
            raise ValueError("Model hasn't been built yet")
        
        # Create a string buffer to store the summary
        from io import StringIO
        summary_buffer = StringIO()
        
        # Save the summary to our buffer
        self.model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        
        return summary_buffer.getvalue()