import pytest
import tensorflow as tf
from src.model_builder import NeuralNetworkBuilder

@pytest.fixture
def model_builder():
    """Create a model builder instance for testing."""
    return NeuralNetworkBuilder(input_dim=10)

def test_model_builder_initialization():
    """Test ModelBuilder initialization."""
    builder = NeuralNetworkBuilder(input_dim=10)
    assert builder.input_dim == 10
    assert builder.model is None

def test_build_simple_model(model_builder):
    """Test building a simple model with basic configuration."""
    model = model_builder.build_model(
        hidden_layers=[32, 16],
        dropout_rates=None,
        use_batch_norm=False
    )
    
    # Check model structure
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 3  # Input dense + hidden dense + output
    assert model.layers[0].units == 32
    assert model.layers[1].units == 16
    assert model.layers[-1].units == 1

def test_build_complex_model(model_builder):
    """Test building a model with all features enabled."""
    model = model_builder.build_model(
        hidden_layers=[32, 16, 8],
        dropout_rates=[0.3, 0.2, 0.1],
        use_batch_norm=True,
        kernel_regularizer={'l1': 1e-5, 'l2': 1e-4}
    )
    
    # Count layers by type
    dense_layers = len([l for l in model.layers if isinstance(l, tf.keras.layers.Dense)])
    dropout_layers = len([l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)])
    batch_norm_layers = len([l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)])
    
    assert dense_layers == 4  # 3 hidden + output
    assert dropout_layers == 3
    assert batch_norm_layers == 3

def test_compile_model(model_builder):
    """Test model compilation with different optimizers and metrics."""
    model = model_builder.build_model(hidden_layers=[32, 16])
    model_builder.compile_model(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    assert model_builder.model.optimizer.__class__.__name__ == 'Adam'
    assert len(model_builder.model.metrics) == 2

def test_compile_model_before_build():
    """Test proper error handling when trying to compile before building."""
    builder = NeuralNetworkBuilder(input_dim=10)
    with pytest.raises(ValueError):
        builder.compile_model()

def test_create_default_model():
    """Test the default model creation utility method."""
    model = NeuralNetworkBuilder.create_default_model(input_dim=10)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 1)

def test_get_model_summary(model_builder):
    """Test model summary generation."""
    model_builder.build_model(hidden_layers=[32, 16])
    summary = model_builder.get_model_summary()
    
    assert isinstance(summary, str)
    assert 'Layer (type)' in summary
    assert 'dense' in summary.lower()
    
def test_invalid_hidden_layers(model_builder):
    """Test error handling for invalid layer configurations."""
    with pytest.raises(ValueError):
        model_builder.build_model(hidden_layers=[])