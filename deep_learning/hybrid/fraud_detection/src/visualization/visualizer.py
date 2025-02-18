import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

class FraudVisualization:
    def __init__(self, output_dir='output'):
        """Initialize visualization class with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_som_heatmap(self, som_detector, X, y):
        """Plot SOM heatmap with detected patterns."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Get SOM feature map and winning nodes
            distance_map = som_detector.get_feature_map()
            winners = som_detector.get_winning_nodes(X)
            
            # Plot the heatmap
            plt.pcolor(distance_map.T, cmap='bone_r')
            plt.colorbar()
            
            # Plot markers for genuine and fraudulent cases
            frauds = (y == 1)
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
            plt.savefig(os.path.join(self.output_dir, 'som_heatmap.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting SOM heatmap: {str(e)}")
            raise
    
    def plot_fraud_distribution(self, predictions, X_original):
        """Plot distribution of features for fraudulent vs non-fraudulent cases."""
        try:
            frauds = (predictions == 1)
            n_features = X_original.shape[1]
            
            plt.figure(figsize=(15, 10))
            for i in range(n_features):
                plt.subplot(3, 5, i + 1)
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
                plt.title(f'Feature {i+1}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'fraud_distribution.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting fraud distribution: {str(e)}")
            raise
    
    def plot_feature_importance(self, ann_classifier, feature_names, X=None):
        """Plot feature importance from ANN model."""
        try:
            importance = ann_classifier.get_feature_importance(X)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=importance,
                y=feature_names,
                palette='viridis'
            )
            plt.title('Feature Importance in Fraud Detection')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_model_metrics(self, y_true, y_pred):
        """Plot confusion matrix and print classification report."""
        try:
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
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
            plt.close()
            
            # Print classification report
            report = classification_report(y_true, y_pred)
            logger.info(f"\nClassification Report:\n{report}")