import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from typing import Dict, Tuple, Optional
import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A comprehensive model evaluation and visualization tool.
    
    This class provides methods for:
    - Performance metrics calculation
    - Learning curves visualization
    - ROC and PR curves
    - Confusion matrix visualization
    - Model prediction analysis
    """
    
    def __init__(self, model: tf.keras.Model, save_dir: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained tensorflow model
            save_dir: Directory to save evaluation results and plots
        """
        self.model = model
        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / 'evaluation_results'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_history(self, history: tf.keras.callbacks.History) -> None:
        """
        Plot training and validation metrics over epochs.
        
        Args:
            history: Training history from model.fit()
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot and save confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (rounded predictions)
        """
        cm = confusion_matrix(y_true, y_pred.round())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Plot ROC curve and calculate AUC.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            float: Area under the ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / 'roc_curve.png')
        plt.close()
        
        return roc_auc
        
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Plot precision-recall curve and calculate average precision.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            float: Average precision score
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(self.save_dir / 'precision_recall_curve.png')
        plt.close()
        
        return avg_precision
    
    def evaluate_model(self, 
                    X_test: np.ndarray, 
                    y_test: np.ndarray,
                    history: Optional[tf.keras.callbacks.History] = None) -> Dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: True labels
            history: Optional training history for learning curves
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = y_pred_proba.round()
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': self.plot_roc_curve(y_test, y_pred_proba),
            'average_precision': self.plot_precision_recall_curve(y_test, y_pred_proba)
        }
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot training history if provided
        if history:
            self.plot_training_history(history)
            
        # Save metrics to file
        with open(self.save_dir / 'evaluation_metrics.txt', 'w') as f:
            f.write("Classification Report:\n")
            f.write(metrics['classification_report'])
            f.write(f"\nROC AUC Score: {metrics['roc_auc']:.4f}")
            f.write(f"\nAverage Precision Score: {metrics['average_precision']:.4f}")
            
        logger.info("Model evaluation completed successfully")
        return metrics
    
    def analyze_predictions(self, 
                        X_test: np.ndarray, 
                        y_test: np.ndarray,
                        feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze model predictions in detail.
        
        Args:
            X_test: Test features
            y_test: True labels
            feature_names: Optional list of feature names
            
        Returns:
            pd.DataFrame: DataFrame containing prediction analysis
        """
        y_pred_proba = self.model.predict(X_test)
        y_pred = y_pred_proba.round()
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'True_Label': y_test.flatten(),
            'Predicted_Proba': y_pred_proba.flatten(),
            'Predicted_Label': y_pred.flatten(),
            'Correct_Prediction': y_test.flatten() == y_pred.flatten()
        })
        
        # Add feature values if names are provided
        if feature_names:
            for i, name in enumerate(feature_names):
                analysis_df[name] = X_test[:, i]
        
        # Save analysis to CSV
        analysis_df.to_csv(self.save_dir / 'prediction_analysis.csv', index=False)
        
        return analysis_df
