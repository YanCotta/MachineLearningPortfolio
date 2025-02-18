import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FraudVisualization:
    def __init__(self, output_dir='output'):
        """Initialize visualization class with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_som_heatmap(self, som_detector, X, y):
        """Plot SOM heatmap with detected patterns."""
        try:
            if not som_detector._trained:
                raise RuntimeError("SOM must be trained before plotting")
            
            plt.figure(figsize=(12, 8))
            
            # Get SOM feature map and winning nodes
            distance_map = som_detector.get_feature_map()
            winners = som_detector.get_winning_nodes(X)
            
            if not np.all(np.isfinite(distance_map)):
                raise ValueError("Invalid values in SOM feature map")
            
            # Plot the heatmap
            plt.pcolor(distance_map.T, cmap='bone_r')
            plt.colorbar()
            
            # Plot markers for genuine and fraudulent cases
            frauds = (y == 1)
            if np.any(frauds):  # Only plot if we have fraud cases
                plt.plot(
                    winners[~frauds, 0] + 0.5,
                    winners[~frauds, 1] + 0.5,
                    'go', marker='o', label='Genuine',
                    alpha=0.7
                )
                plt.plot(
                    winners[frauds, 0] + 0.5,
                    winners[frauds, 1] + 0.5,
                    'ro', marker='s', label='Fraud',
                    alpha=0.7
                )
            
            plt.title('SOM Fraud Detection Map')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'som_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("SOM heatmap saved successfully")
            
        except Exception as e:
            logger.error(f"Error plotting SOM heatmap: {str(e)}")
            raise
    
    def plot_fraud_distribution(self, predictions, X_original, feature_names=None):
        """Plot distribution of features for fraudulent vs non-fraudulent cases."""
        try:
            if len(predictions) != len(X_original):
                raise ValueError("Length mismatch between predictions and features")
            
            frauds = (predictions == 1)
            if not np.any(frauds):
                logger.warning("No fraud cases detected for distribution plot")
                return
                
            n_features = X_original.shape[1]
            n_cols = min(5, n_features)
            n_rows = (n_features - 1) // n_cols + 1
            
            plt.figure(figsize=(4*n_cols, 3*n_rows))
            for i in range(n_features):
                plt.subplot(n_rows, n_cols, i + 1)
                
                # Check for invalid values
                if not np.all(np.isfinite(X_original[:, i])):
                    logger.warning(f"Invalid values in feature {i}, skipping plot")
                    continue
                
                sns.kdeplot(
                    data=X_original[~frauds, i],
                    label='Genuine',
                    alpha=0.5
                )
                sns.kdeplot(
                    data=X_original[frauds, i],
                    label='Fraud',
                    alpha=0.5
                )
                
                title = f'Feature {feature_names[i]}' if feature_names else f'Feature {i+1}'
                plt.title(title, fontsize=10)
                if i % n_cols == 0:  # Only show legend for leftmost plots
                    plt.legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'fraud_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Fraud distribution plot saved successfully")
            
        except Exception as e:
            logger.error(f"Error plotting fraud distribution: {str(e)}")
            raise
    
    def plot_feature_importance(self, ann_classifier, feature_names, X):
        """Plot feature importance from ANN model."""
        try:
            if not ann_classifier._trained:
                raise RuntimeError("ANN must be trained before plotting importance")
                
            importance = ann_classifier.get_feature_importance(X)
            
            if len(importance) != len(feature_names):
                raise ValueError("Length mismatch between importance scores and feature names")
            
            # Sort features by importance
            idx = np.argsort(importance)
            sorted_features = np.array(feature_names)[idx]
            sorted_importance = importance[idx]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=sorted_importance,
                y=sorted_features,
                palette='viridis'
            )
            plt.title('Feature Importance in Fraud Detection')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature importance plot saved successfully")
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_model_metrics(self, y_true, y_pred):
        """Plot confusion matrix and print classification report."""
        try:
            if len(y_true) != len(y_pred):
                raise ValueError("Length mismatch between true and predicted labels")
                
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues'
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and log metrics
            report = classification_report(y_true, y_pred)
            logger.info(f"\nClassification Report:\n{report}")
            
            # Log confusion matrix details
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            logger.info(f"True Negatives: {tn}")
            logger.info(f"False Positives: {fp}")
            logger.info(f"False Negatives: {fn}")
            logger.info(f"True Positives: {tp}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            
        except Exception as e:
            logger.error(f"Error plotting model metrics: {str(e)}")
            raise