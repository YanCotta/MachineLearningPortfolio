import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various regression metrics for model evaluation
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    logger.info("Evaluation metrics calculated:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    return metrics

def evaluate_predictions(actual: np.ndarray, 
                       predicted: np.ndarray,
                       scaler: object) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate predictions in both scaled and original formats
    
    Args:
        actual: Actual scaled values
        predicted: Predicted scaled values
        scaler: Fitted scaler object for inverse transformation
        
    Returns:
        Tuple of dictionaries containing metrics for scaled and original data
    """
    # Calculate metrics on scaled data
    scaled_metrics = calculate_metrics(actual, predicted)
    
    # Transform back to original scale
    actual_orig = scaler.inverse_transform(actual.reshape(-1, 1))
    predicted_orig = scaler.inverse_transform(predicted.reshape(-1, 1))
    
    # Calculate metrics on original scale
    original_metrics = calculate_metrics(actual_orig, predicted_orig)
    
    return scaled_metrics, original_metrics