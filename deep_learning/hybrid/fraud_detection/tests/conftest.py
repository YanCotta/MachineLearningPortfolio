import pytest
import numpy as np
import tempfile
from pathlib import Path

@pytest.fixture
def test_data():
    """Provide test data for models."""
    input_dim = 10
    n_samples = 100
    X = np.random.rand(n_samples, input_dim)
    y = np.random.randint(0, 2, n_samples)
    feature_names = [f'Feature_{i}' for i in range(input_dim)]
    return X, y, feature_names, input_dim

@pytest.fixture
def temp_output_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)