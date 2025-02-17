import pytest
import numpy as np
from src.model import StockPredictor
import tensorflow as tf
import os

@pytest.fixture
def sample_data():
    # Create sample data for testing
    sequence_length = 60
    n_samples = 100
    
    X = np.random.random((n_samples, sequence_length))
    y = np.random.random(n_samples)
    
    return X, y

def test_model_initialization():
    sequence_length = 60
    model = StockPredictor(sequence_length)
    
    # Check model structure
    assert isinstance(model.model, tf.keras.Model)
    assert model.sequence_length == sequence_length
    
    # Check input shape
    assert model.model.input_shape == (None, sequence_length, 1)

def test_model_training(sample_data):
    X, y = sample_data
    sequence_length = X.shape[1]
    
    model = StockPredictor(sequence_length)
    history = model.train(X, y, epochs=2, batch_size=32)
    
    # Check training history
    assert 'loss' in history.history
    assert len(history.history['loss']) == 2

def test_model_prediction(sample_data):
    X, _ = sample_data
    sequence_length = X.shape[1]
    
    model = StockPredictor(sequence_length)
    predictions = model.predict(X)
    
    # Check predictions shape and values
    assert predictions.shape == (len(X), 1)
    assert np.all(np.isfinite(predictions))  # No NaN or infinite values

def test_model_save_load(sample_data, tmpdir):
    X, y = sample_data
    sequence_length = X.shape[1]
    
    # Create and train model
    model = StockPredictor(sequence_length)
    model.train(X, y, epochs=1)
    
    # Save weights
    weights_path = os.path.join(tmpdir, 'test_weights.h5')
    model.save_weights(weights_path)
    
    # Create new model and load weights
    new_model = StockPredictor(sequence_length)
    new_model.load_weights(weights_path)
    
    # Compare predictions
    pred1 = model.predict(X)
    pred2 = new_model.predict(X)
    np.testing.assert_array_almost_equal(pred1, pred2)