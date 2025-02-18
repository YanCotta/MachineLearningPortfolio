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
import wandb
import mlflow
import mlflow.tensorflow
import json

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and experiment management."""

    def __init__(self, 
                model,
                experiment_name: Optional[str] = None,
                checkpoint_dir: str = "checkpoints",
                use_wandb: bool = True,
                use_mlflow: bool = True):
        """
        Initialize the trainer with experiment tracking.

        Args:
            model: The CNN model instance
            experiment_name: Optional name for the experiment
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for tracking
            use_mlflow: Whether to use MLflow for tracking
        """
        self.model = model
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
        
        # Initialize experiment tracking
        if use_wandb:
            wandb.init(project="cnn_classifier", name=self.experiment_name)
            wandb.config.update({
                "architecture": "CNN",
                "dataset": "binary_classification",
                "img_size": model.img_size,
                "optimizer": model.model.optimizer.get_config()
            })
        
        if use_mlflow:
            mlflow.set_experiment("cnn_classifier")
            mlflow.tensorflow.autolog()

    def train(self,
            train_data: tf.keras.preprocessing.image.DirectoryIterator,
            validation_data: tf.keras.preprocessing.image.DirectoryIterator,
            epochs: int = 25,
            initial_epoch: int = 0) -> tf.keras.callbacks.History:
        """
        Train the model with comprehensive monitoring.

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
            histogram_freq=1,
            profile_batch='500,520'  # Profile performance
        )

        # Get model callbacks
        callbacks = self.model.get_callbacks() + [
            tensorboard_callback,
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._log_metrics(epoch, logs)
            )
        ]

        # Train with mixed precision if available
        if tf.config.list_physical_devices('GPU'):
            logger.info("Training with mixed precision on GPU")
        else:
            logger.warning("No GPU detected, training on CPU")

        # Train the model
        logger.info(f"Starting training for {epochs} epochs...")
        try:
            self.history = self.model.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                workers=4,  # Parallel data loading
                use_multiprocessing=True
            )

            # Save training history
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f)

            # Save final model
            final_model_path = self.checkpoint_dir / "final_model.h5"
            self.model.save_model(str(final_model_path))
            
            # Log final metrics
            self._log_final_metrics()
            
            logger.info(f"Training completed. Final model saved to {final_model_path}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

        return self.history

    def _log_metrics(self, epoch: int, logs: dict) -> None:
        """Log metrics to tracking platforms."""
        if wandb.run is not None:
            wandb.log(logs)

    def _log_final_metrics(self) -> None:
        """Log final model metrics and artifacts."""
        if wandb.run is not None:
            wandb.log({
                "final_val_accuracy": self.history.history['val_accuracy'][-1],
                "final_val_loss": self.history.history['val_loss'][-1],
                "best_val_accuracy": max(self.history.history['val_accuracy']),
                "best_val_loss": min(self.history.history['val_loss'])
            })
            wandb.save(str(self.checkpoint_dir / "final_model.h5"))

    def resume_training(self,
                    train_data: tf.keras.preprocessing.image.DirectoryIterator,
                    validation_data: tf.keras.preprocessing.image.DirectoryIterator,
                    checkpoint_path: str,
                    additional_epochs: int = 10) -> tf.keras.callbacks.History:
        """
        Resume training from a checkpoint with state preservation.

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
        """Extract epoch number from checkpoint filename."""
        try:
            filename = Path(checkpoint_path).stem
            if 'epoch' in filename:
                return int(filename.split('epoch')[-1].split('_')[0])
        except ValueError:
            pass
        return 0