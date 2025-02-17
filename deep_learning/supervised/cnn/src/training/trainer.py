"""
Model training and experiment management.

This module handles the training process, including experiment tracking
and model checkpointing.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and experiment management."""

    def __init__(self, 
                model,
                experiment_name: Optional[str] = None,
                checkpoint_dir: str = "checkpoints"):
        """
        Initialize the trainer.

        Args:
            model: The CNN model instance
            experiment_name: Optional name for the experiment
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = None

    def train(self,
            train_data: tf.keras.preprocessing.image.DirectoryIterator,
            validation_data: tf.keras.preprocessing.image.DirectoryIterator,
            epochs: int = 25,
            initial_epoch: int = 0) -> tf.keras.callbacks.History:
        """
        Train the model with the given data.

        Args:
            train_data: Training data iterator
            validation_data: Validation data iterator
            epochs: Number of epochs to train
            initial_epoch: Epoch to start training from

        Returns:
            Training history
        """
        # Set up TensorBoard logging
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )

        # Get model callbacks
        callbacks = self.model.get_callbacks() + [tensorboard_callback]

        # Train the model
        logger.info(f"Starting training for {epochs} epochs...")
        self.history = self.model.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )

        # Save final model
        final_model_path = self.checkpoint_dir / "final_model.h5"
        self.model.save_model(str(final_model_path))
        logger.info(f"Training completed. Final model saved to {final_model_path}")

        return self.history

    def resume_training(self,
                    train_data: tf.keras.preprocessing.image.DirectoryIterator,
                    validation_data: tf.keras.preprocessing.image.DirectoryIterator,
                    checkpoint_path: str,
                    additional_epochs: int = 10) -> tf.keras.callbacks.History:
        """
        Resume training from a checkpoint.

        Args:
            train_data: Training data iterator
            validation_data: Validation data iterator
            checkpoint_path: Path to the checkpoint to resume from
            additional_epochs: Number of additional epochs to train

        Returns:
            Training history
        """
        # Load the checkpoint
        self.model.load_model(checkpoint_path)
        logger.info(f"Resumed training from checkpoint: {checkpoint_path}")

        # Get the initial epoch from the checkpoint filename
        initial_epoch = self._get_epoch_from_checkpoint(checkpoint_path)

        # Continue training
        return self.train(
            train_data,
            validation_data,
            epochs=initial_epoch + additional_epochs,
            initial_epoch=initial_epoch
        )

    @staticmethod
    def _get_epoch_from_checkpoint(checkpoint_path: str) -> int:
        """
        Extract epoch number from checkpoint filename.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Epoch number
        """
        try:
            filename = Path(checkpoint_path).stem
            if 'epoch' in filename:
                return int(filename.split('epoch')[-1].split('_')[0])
        except ValueError:
            pass
        return 0