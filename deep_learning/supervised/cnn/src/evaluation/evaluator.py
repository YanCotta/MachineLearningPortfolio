"""
Model evaluation and visualization utilities.

This module provides comprehensive evaluation metrics and visualization
tools for analyzing model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and performance visualization."""

    def __init__(self):
        """Initialize the evaluator."""
        self.history = None
        self.class_names = ['cat', 'dog']  # Binary classification

    def set_training_history(self, history: tf.keras.callbacks.History) -> None:
        """
        Set the training history for visualization.

        Args:
            history: Keras training history object
        """
        self.history = history

    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics history.

        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training History', size=16)

        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'])
        axes[0, 0].plot(self.history.history['val_accuracy'])
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(['Train', 'Validation'])

        # Plot loss
        axes[0, 1].plot(self.history.history['loss'])
        axes[0, 1].plot(self.history.history['val_loss'])
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(['Train', 'Validation'])

        # Plot AUC
        axes[1, 0].plot(self.history.history['auc'])
        axes[1, 0].plot(self.history.history['val_auc'])
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend(['Train', 'Validation'])

        # Plot learning rate if available
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

    def evaluate_model(self, 
                      model: tf.keras.Model, 
                      test_data: tf.keras.preprocessing.image.DirectoryIterator) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.

        Args:
            model: Trained Keras model
            test_data: Test dataset iterator

        Returns:
            Dictionary containing evaluation metrics
        """
        # Reset the test data generator
        test_data.reset()
        
        # Get predictions
        y_pred = model.predict(test_data)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Get true labels
        y_true = test_data.classes

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Get classification report
        report = classification_report(y_true, y_pred_classes, 
                                     target_names=self.class_names, 
                                     output_dict=True)

        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'true_labels': y_true
        }

    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray, 
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix using seaborn.

        Args:
            confusion_matrix: Computed confusion matrix
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()

    def visualize_predictions(self, 
                            model: tf.keras.Model,
                            test_data: tf.keras.preprocessing.image.DirectoryIterator,
                            num_samples: int = 9,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize model predictions on sample images.

        Args:
            model: Trained Keras model
            test_data: Test dataset iterator
            num_samples: Number of samples to visualize
            save_path: Optional path to save the plot
        """
        test_data.reset()
        images, labels = next(test_data)
        predictions = model.predict(images)

        fig = plt.figure(figsize=(15, 15))
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            pred_class = self.class_names[int(predictions[i] > 0.5)]
            true_class = self.class_names[int(labels[i])]
            color = 'green' if pred_class == true_class else 'red'
            plt.title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
            plt.axis('off')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Predictions visualization saved to {save_path}")
        else:
            plt.show()