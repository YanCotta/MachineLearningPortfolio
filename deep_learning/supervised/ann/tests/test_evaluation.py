import pytest
import numpy as np
import tensorflow as tf
from src.evaluation import ModelEvaluator
import os
import tempfile

@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(10,))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture
def test_data():
    """Create test data for evaluation."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def dummy_history():
    """Create a dummy training history."""
    class DummyHistory:
        def __init__(self):
            self.history = {
                'accuracy': [0.7, 0.8, 0.85],
                'val_accuracy': [0.65, 0.75, 0.8],
                'loss': [0.5, 0.3, 0.2],
                'val_loss': [0.6, 0.4, 0.3]
            }
    return DummyHistory()

def test_evaluator_initialization(dummy_model):
    """Test ModelEvaluator initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(dummy_model, save_dir=tmpdir)
        assert evaluator.model is not None
        assert os.path.exists(evaluator.save_dir)

def test_plot_training_history(dummy_model, dummy_history):
    """Test training history plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(dummy_model, save_dir=tmpdir)
        evaluator.plot_training_history(dummy_history)
        assert os.path.exists(os.path.join(tmpdir, 'training_history.png'))

def test_plot_confusion_matrix(dummy_model, test_data):
    """Test confusion matrix plotting."""
    X, y = test_data
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(dummy_model, save_dir=tmpdir)
        y_pred = dummy_model.predict(X)
        evaluator.plot_confusion_matrix(y, y_pred.round())
        assert os.path.exists(os.path.join(tmpdir, 'confusion_matrix.png'))

def test_plot_roc_curve(dummy_model, test_data):
    """Test ROC curve plotting."""
    X, y = test_data
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(dummy_model, save_dir=tmpdir)
        y_pred = dummy_model.predict(X)
        auc_score = evaluator.plot_roc_curve(y, y_pred)
        assert isinstance(auc_score, float)
        assert 0 <= auc_score <= 1
        assert os.path.exists(os.path.join(tmpdir, 'roc_curve.png'))

def test_evaluate_model(dummy_model, test_data, dummy_history):
    """Test complete model evaluation."""
    X, y = test_data
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(dummy_model, save_dir=tmpdir)
        metrics = evaluator.evaluate_model(X, y, dummy_history)
        
        assert isinstance(metrics, dict)
        assert 'classification_report' in metrics
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        
        assert os.path.exists(os.path.join(tmpdir, 'evaluation_metrics.txt'))
        assert os.path.exists(os.path.join(tmpdir, 'confusion_matrix.png'))
        assert os.path.exists(os.path.join(tmpdir, 'roc_curve.png'))
        assert os.path.exists(os.path.join(tmpdir, 'precision_recall_curve.png'))

def test_analyze_predictions(dummy_model, test_data):
    """Test prediction analysis functionality."""
    X, y = test_data
    feature_names = [f'feature_{i}' for i in range(10)]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(dummy_model, save_dir=tmpdir)
        analysis = evaluator.analyze_predictions(X, y, feature_names)
        
        assert len(analysis) == len(y)
        assert all(col in analysis.columns for col in ['True_Label', 'Predicted_Proba', 'Predicted_Label'])
        assert os.path.exists(os.path.join(tmpdir, 'prediction_analysis.csv'))