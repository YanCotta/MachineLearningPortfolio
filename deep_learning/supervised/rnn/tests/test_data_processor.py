import pytest
import numpy as np
import pandas as pd
from src.data_processor import DataProcessor
import os
from datetime import datetime, timedelta

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

@pytest.fixture
def sample_stock_data():
    """Create sample stock market data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(150, 250, 100),
        'Low': np.random.uniform(90, 180, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

@pytest.fixture
def data_processor():
    """Create DataProcessor instance for testing."""
    return DataProcessor()

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

def test_data_loading(data_processor, tmp_path):
    """Test data loading and validation."""
    # Create test CSV file
    df = pd.DataFrame({
        'Date': ['2020-01-01', '2020-01-02'],
        'Open': [100, 101],
        'High': [105, 106],
        'Low': [98, 99],
        'Close': [102, 103],
        'Volume': [1000, 1100]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Test loading
    loaded_df = data_processor.load_data(str(csv_path))
    assert isinstance(loaded_df.index, pd.DatetimeIndex)
    assert all(col in loaded_df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_sequence_preparation(data_processor):
    """Test sequence preparation functionality."""
    # Create sample data
    data = np.array([[1], [2], [3], [4], [5]])
    sequence_length = 2
    
    X, y = data_processor.prepare_sequences(data, sequence_length)
    
    assert X.shape == (3, 2, 1)  # 3 sequences of length 2 with 1 feature
    assert y.shape == (3,)  # 3 target values
    np.testing.assert_array_equal(X[0], [[1], [2]])
    np.testing.assert_array_equal(y, [3, 4, 5])

def test_data_preparation(data_processor, sample_stock_data):
    """Test end-to-end data preparation."""
    sequence_length = 10
    features = ['Close', 'Volume']
    
    X_train, X_test, y_train, y_test = data_processor.prepare_data(
        sample_stock_data,
        sequence_length=sequence_length,
        train_split=0.8,
        features=features
    )
    
    # Check shapes
    expected_total_sequences = len(sample_stock_data) - sequence_length
    expected_train_size = int(expected_total_sequences * 0.8)
    expected_test_size = expected_total_sequences - expected_train_size
    
    assert X_train.shape[1:] == (sequence_length, len(features))
    assert X_test.shape[1:] == (sequence_length, len(features))
    assert len(X_train) == expected_train_size
    assert len(X_test) == expected_test_size

def test_technical_indicators(data_processor, sample_stock_data):
    """Test technical indicators calculation."""
    df_with_indicators = data_processor.add_technical_indicators(sample_stock_data)
    
    # Check if all indicators are present
    expected_indicators = ['MA7', 'MA20', 'RSI', 'MACD', 'Signal_Line', 
                         'BB_middle', 'BB_upper', 'BB_lower']
    assert all(indicator in df_with_indicators.columns for indicator in expected_indicators)
    
    # Check if indicators are calculated correctly
    assert df_with_indicators['MA7'].iloc[7:].notna().all()  # MA7 should have values after 7 days
    assert df_with_indicators['MA20'].iloc[20:].notna().all()  # MA20 should have values after 20 days
    assert (df_with_indicators['RSI'] >= 0).all() and (df_with_indicators['RSI'] <= 100).all()

def test_inverse_transform(data_processor):
    """Test inverse transform functionality."""
    # Create sample scaled data
    original_data = np.array([[100], [150], [200]])
    scaled_data = data_processor.scaler.fit_transform(original_data)
    
    # Test inverse transform
    restored_data = data_processor.inverse_transform(scaled_data)
    np.testing.assert_array_almost_equal(original_data, restored_data)

def test_missing_values_handling(data_processor, sample_stock_data):
    """Test handling of missing values."""
    # Introduce some NaN values
    sample_stock_data.iloc[5:10, :] = np.nan
    
    processed_df = data_processor.load_data(sample_stock_data)
    
    # Check if NaN values are handled
    assert not processed_df.isnull().any().any()

def test_invalid_data_loading(data_processor, tmp_path):
    """Test handling of invalid data loading."""
    # Create invalid CSV file
    df = pd.DataFrame({
        'Date': ['2020-01-01'],
        'Price': [100]  # Missing required columns
    })
    csv_path = tmp_path / "invalid_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Test that loading invalid data raises error
    with pytest.raises(ValueError):
        data_processor.load_data(str(csv_path))