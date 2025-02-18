"""
CNN model architecture and configuration.

This module implements a modern CNN architecture for binary image classification,
with advanced features including:
- ResNet-style skip connections
- Advanced regularization techniques
- Configurable architecture parameters
- Multi-GPU support
- Mixed precision training
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_addons.layers import StochasticDepth
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CNNModel:
    """Implements a modern CNN architecture for binary image classification."""

    def __init__(self, 
                img_size: Tuple[int, int] = (64, 64),
                num_filters: int = 32,
                num_blocks: int = 3,
                use_residual: bool = True,
                dropout_rate: float = 0.3,
                l2_reg: float = 1e-4,
                use_mixed_precision: bool = True):
        """
        Initialize the CNN model with advanced configuration options.

        Args:
            img_size: Tuple of (height, width) for input images
            num_filters: Initial number of filters (doubled in each block)
            num_blocks: Number of residual/convolutional blocks
            use_residual: Whether to use residual connections
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            use_mixed_precision: Whether to use mixed precision training
        """
        self.img_size = img_size
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Enabled mixed precision training")

    def _create_conv_block(self, inputs: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
        """Create a convolutional block with batch norm and residual connection."""
        x = layers.Conv2D(
            filters, 3, strides=strides, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(
            filters, 3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)

        if self.use_residual:
            if strides > 1 or inputs.shape[-1] != filters:
                inputs = layers.Conv2D(filters, 1, strides=strides, padding='same')(inputs)
            x = layers.Add()([x, inputs])

        x = layers.Activation('relu')(x)
        x = layers.SpatialDropout2D(self.dropout_rate)(x)
        return x

    def build(self) -> None:
        """
        Build the CNN architecture with modern best practices.
        """
        inputs = layers.Input(shape=[*self.img_size, 3])
        
        # Initial processing
        x = layers.Conv2D(
            self.num_filters, 7, strides=2, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # Convolutional blocks with increasing filters
        filters = self.num_filters
        for i in range(self.num_blocks):
            x = self._create_conv_block(x, filters)
            x = self._create_conv_block(x, filters, strides=2)
            filters *= 2

        # Global pooling and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(
            512, 
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with proper initialization
        outputs = layers.Dense(
            1, 
            activation='sigmoid',
            kernel_initializer='glorot_normal'
        )(x)

        self.model = models.Model(inputs, outputs)
        
        # Use AMP optimizer wrapper for mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score'),
            ]
        )
        
        logger.info("Model built successfully")
        self.model.summary()

    def get_callbacks(self, monitor: str = 'val_loss') -> List[tf.keras.callbacks.Callback]:
        """
        Get enhanced training callbacks for improved training process.

        Args:
            monitor: Metric to monitor for early stopping and checkpoints

        Returns:
            List of Keras callbacks for training
        """
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor=monitor,
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            ),
            tf.keras.callbacks.CSVLogger(
                'training_log.csv',
                separator=',',
                append=False
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