import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation import ModelEvaluator
from pathlib import Path

@pytest.fixture
def evaluator(tmp_path):
    """Create ModelEvaluator instance with temporary directory."""
    return ModelEvaluator(output_dir=str(tmp_path))

@pytest.fixture
def sample_data():
    """Generate sample prediction data."""
    np.random.seed(42)
    y_true = np.sin(np.linspace(0, 10, 100)) * 100 + 500
    y_pred = y_true + np.random.normal(0, 5, 100)  # Add some noise
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    return y_true, y_pred, dates

def test_metrics_calculation(evaluator, sample_data):
    """Test calculation of evaluation metrics."""
    y_true, y_pred, _ = sample_data
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    assert isinstance(metrics, dict)
    required_metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
    assert all(metric in metrics for metric in required_metrics)
    assert metrics['R2'] <= 1.0  # R² should be <= 1
    assert metrics['MAPE'] >= 0  # MAPE should be positive
    assert metrics['RMSE'] >= 0  # RMSE should be positive

def test_prediction_plotting(evaluator, sample_data, tmp_path):
    """Test prediction visualization functionality."""
    y_true, y_pred, _ = sample_data
    save_path = 'test_plot.png'
    
    evaluator.plot_predictions(y_true, y_pred, save_path=save_path)
    assert (evaluator.output_dir / save_path).exists()

def test_training_history_plotting(evaluator, tmp_path):
    """Test training history visualization."""
    history = {
        'loss': [0.1, 0.05, 0.03],
        'val_loss': [0.12, 0.07, 0.04],
        'mae': [0.2, 0.15, 0.1],
        'val_mae': [0.22, 0.17, 0.12]
    }
    save_path = 'history_plot.png'
    
    evaluator.plot_training_history(history, save_path=save_path)
    assert (evaluator.output_dir / save_path).exists()

def test_prediction_analysis(evaluator, sample_data):
    """Test comprehensive prediction analysis."""
    y_true, y_pred, dates = sample_data
    
    analysis = evaluator.analyze_predictions(y_true, y_pred, dates)
    
    assert 'metrics' in analysis
    assert 'error_stats' in analysis
    
    error_stats = analysis['error_stats']
    assert 'direction_accuracy' in error_stats
    assert 0 <= error_stats['direction_accuracy'] <= 100
    
    # Check error distribution plot
    assert (evaluator.output_dir / 'error_distribution.png').exists()

def test_report_generation(evaluator, sample_data):
    """Test evaluation report generation."""
    y_true, y_pred, dates = sample_data
    
    analysis = evaluator.analyze_predictions(y_true, y_pred, dates)
    report = evaluator.generate_report(analysis)
    
    assert isinstance(report, str)
    assert "Performance Metrics:" in report
    assert "Error Statistics:" in report
    
    # Check if report file is created
    assert (evaluator.output_dir / 'evaluation_report.txt').exists()
    
    # Verify report content
    with open(evaluator.output_dir / 'evaluation_report.txt', 'r') as f:
        saved_report = f.read()
    assert saved_report == report

def test_direction_accuracy(evaluator, sample_data):
    """Test direction prediction accuracy calculation."""
    # Create data with known directional changes
    y_true = np.array([1, 2, 1, 3, 2])
    y_pred = np.array([1, 2.1, 0.9, 3.1, 1.9])
    
    analysis = evaluator.analyze_predictions(y_true, y_pred)
    direction_accuracy = analysis['error_stats']['direction_accuracy']
    
    # All directional changes are correctly predicted
    assert np.isclose(direction_accuracy, 100.0)

def test_error_statistics(evaluator):
    """Test error statistics calculation with controlled data."""
    y_true = np.array([100, 110, 120, 130, 140])
    y_pred = np.array([102, 111, 118, 131, 142])
    
    analysis = evaluator.analyze_predictions(y_true, y_pred)
    error_stats = analysis['error_stats']
    
    assert 'mean_error' in error_stats
    assert 'std_error' in error_stats
    assert 'max_error' in error_stats
    
    # Mean error should be close to the actual mean difference
    expected_mean_error = np.mean(y_true - y_pred)
    assert np.isclose(error_stats['mean_error'], expected_mean_error)

def test_output_directory_creation(tmp_path):
    """Test automatic creation of output directory."""
    output_dir = tmp_path / "new_eval_dir"
    evaluator = ModelEvaluator(output_dir=str(output_dir))
    assert output_dir.exists()

def test_handle_empty_data(evaluator):
    """Test handling of empty data."""
    with pytest.raises(ValueError):
        evaluator.calculate_metrics(np.array([]), np.array([]))