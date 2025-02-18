import pytest
import numpy as np
import tensorflow as tf
from src.model import StockPredictor

@pytest.fixture
def sample_data():
    # Generate sample data for testing
    sequence_length = 10
    n_features = 3
    n_samples = 100
    
    X = np.random.random((n_samples, sequence_length, n_features))
    y = np.random.random((n_samples, 1))
    
    return X, y, sequence_length, n_features

def test_model_initialization():
    """Test model initialization with different configurations."""
    model = StockPredictor(
        sequence_length=10,
        n_features=3,
        lstm_units=[64, 32],
        dropout_rates=[0.2, 0.2],
        bidirectional=True
    )
    
    assert model.sequence_length == 10
    assert model.n_features == 3
    assert len(model.lstm_units) == 2
    assert model.bidirectional == True
    assert isinstance(model.model, tf.keras.Model)

def test_model_architecture(sample_data):
    """Test model architecture and shape outputs."""
    _, _, sequence_length, n_features = sample_data
    
    model = StockPredictor(
        sequence_length=sequence_length,
        n_features=n_features
    )
    
    # Check input shape
    assert model.model.input_shape == (None, sequence_length, n_features)
    # Check output shape
    assert model.model.output_shape == (None, 1)
    
def test_model_training(sample_data):
    """Test model training functionality."""
    X, y, sequence_length, n_features = sample_data
    
    model = StockPredictor(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=[32, 16],
        dropout_rates=[0.2, 0.2]
    )
    
    # Train for a few epochs
    history = model.train(
        X, y,
        validation_data=(X[:10], y[:10]),
        epochs=2,
        batch_size=32
    )
    
    assert 'loss' in history.history
    assert 'val_loss' in history.history
    assert len(history.history['loss']) == 2

def test_model_prediction(sample_data):
    """Test model prediction functionality."""
    X, _, sequence_length, n_features = sample_data
    
    model = StockPredictor(
        sequence_length=sequence_length,
        n_features=n_features
    )
    
    predictions = model.predict(X)
    assert predictions.shape == (len(X), 1)

def test_model_save_load(sample_data, tmpdir):
    """Test model save and load functionality."""
    X, y, sequence_length, n_features = sample_data
    save_path = tmpdir.mkdir("model_saves")
    
    model = StockPredictor(
        sequence_length=sequence_length,
        n_features=n_features,
        save_path=str(save_path)
    )
    
    # Train the model
    model.train(X, y, epochs=1)
    
    # Save the model
    save_file = str(save_path / "test_model.h5")
    model.save_model(save_file)
    
    # Load the model
    new_model = StockPredictor(
        sequence_length=sequence_length,
        n_features=n_features
    )
    new_model.load_model(save_file)
    
    # Compare predictions
    pred1 = model.predict(X)
    pred2 = new_model.predict(X)
    np.testing.assert_array_almost_equal(pred1, pred2)

def test_callbacks_generation():
    """Test callback generation with different parameters."""
    model = StockPredictor(sequence_length=10, n_features=1)
    callbacks = model.get_callbacks(patience=5)
    
    assert len(callbacks) == 3
    assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)
    assert isinstance(callbacks[1], tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(callbacks[2], tf.keras.callbacks.ReduceLROnPlateau)
    assert callbacks[0].patience == 5