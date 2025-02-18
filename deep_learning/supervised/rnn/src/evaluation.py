import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and performance visualization."""

    def __init__(self, output_dir: str = 'evaluation'):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics
    
    def plot_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        title: str = 'Stock Price Prediction',
                        save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
        
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot training history metrics.
        
        Args:
            history: Keras training history
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
        
    def analyze_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        dates: Optional[pd.DatetimeIndex] = None) -> Dict:
        """
        Perform detailed analysis of predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional dates for time-based analysis
            
        Returns:
            Dictionary containing analysis results
        """
        # Calculate basic metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Direction accuracy
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
        
        # Error distribution
        errors = y_true - y_pred
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'direction_accuracy': direction_accuracy
        }
        
        # Combine all analyses
        analysis = {
            'metrics': metrics,
            'error_stats': error_stats
        }
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / 'error_distribution.png')
        plt.close()
        
        return analysis
    
    def generate_report(self, analysis: Dict) -> str:
        """
        Generate a text report of the evaluation results.
        
        Args:
            analysis: Dictionary containing analysis results
            
        Returns:
            Formatted report string
        """
        report = "Stock Price Prediction Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Performance Metrics
        report += "Performance Metrics:\n"
        report += "-" * 20 + "\n"
        metrics = analysis['metrics']
        for metric, value in metrics.items():
            report += f"{metric}: {value:.4f}\n"
        
        # Error Statistics
        report += "\nError Statistics:\n"
        report += "-" * 20 + "\n"
        error_stats = analysis['error_stats']
        for stat, value in error_stats.items():
            report += f"{stat}: {value:.4f}\n"
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report