import pytest
import numpy as np
import pandas as pd
from src.data_processor import DataProcessor
import os

@pytest.fixture
def sample_data():
    # Create sample CSV files for testing
    train_data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'Close': np.random.random(100) * 100
    })
    
    test_data = pd.DataFrame({
        'Date': pd.date_range(start='2020-04-10', periods=20),
        'Close': np.random.random(20) * 100
    })
    
    train_path = 'test_train.csv'
    test_path = 'test_test.csv'
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    yield train_path, test_path
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_path)

def test_data_processor_initialization(sample_data):
    train_path, test_path = sample_data
    processor = DataProcessor(train_path, test_path)
    assert processor.sequence_length == 60
    assert processor.train_path == train_path
    assert processor.test_path == test_path

def test_data_loading_and_preprocessing(sample_data):
    train_path, test_path = sample_data
    processor = DataProcessor(train_path, test_path)
    
    X_train, y_train, X_test, y_test = processor.load_and_preprocess_data()
    
    # Check shapes
    assert len(X_train.shape) == 2
    assert len(y_train.shape) == 1
    assert len(X_test.shape) == 2
    assert len(y_test.shape) == 1
    
    # Check sequence length
    assert X_train.shape[1] == processor.sequence_length
    assert X_test.shape[1] == processor.sequence_length
    
    # Check scaling
    assert np.all((X_train >= 0) & (X_train <= 1))
    assert np.all((y_train >= 0) & (y_train <= 1))
    assert np.all((X_test >= 0) & (X_test <= 1))
    assert np.all((y_test >= 0) & (y_test <= 1))

def test_inverse_transform(sample_data):
    train_path, test_path = sample_data
    processor = DataProcessor(train_path, test_path)
    
    # Get some scaled data
    X_train, y_train, _, _ = processor.load_and_preprocess_data()
    
    # Test inverse transform
    original_data = processor.inverse_transform(y_train)
    assert original_data.shape[1] == 1
    assert not np.array_equal(original_data, y_train)  # Should be different after inverse transform