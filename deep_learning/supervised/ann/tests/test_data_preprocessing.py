import pytest
import numpy as np
import pandas as pd
from src.data_processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'RowNumber': range(1, 11),
        'CustomerId': range(15634602, 15634612),
        'Surname': ['Smith'] * 10,
        'CreditScore': [619, 608, 502, 699, 850, 645, 822, 376, 501, 684],
        'Geography': ['France', 'Spain', 'France', 'France', 'Spain', 
                    'Spain', 'France', 'Germany', 'France', 'France'],
        'Gender': ['Female', 'Female', 'Female', 'Female', 'Female',
                'Male', 'Male', 'Female', 'Male', 'Male'],
        'Age': [42, 41, 42, 39, 43, 44, 50, 29, 44, 27],
        'Tenure': [2, 1, 8, 1, 2, 8, 7, 4, 4, 2],
        'Balance': [0, 83807.86, 159660.8, 0, 125510.82,
                113755.78, 0, 115046.74, 142051.07, 134603.88],
        'NumOfProducts': [1, 1, 3, 2, 1, 2, 2, 4, 2, 1],
        'HasCrCard': [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        'IsActiveMember': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
        'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63,
                        79084.1, 149756.71, 10062.8, 119346.88,
                        74940.5, 71725.73],
        'Exited': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    })

def test_data_processor_initialization():
    """Test DataProcessor initialization."""
    processor = DataProcessor()
    assert processor.scaler is not None
    assert processor.label_encoder is not None
    assert processor.X_train is None
    assert processor.X_test is None
    assert processor.y_train is None
    assert processor.y_test is None

def test_preprocess_data(sample_data):
    """Test data preprocessing functionality."""
    processor = DataProcessor()
    X, y = processor.preprocess_data(sample_data, 'Exited')
    
    # Check shapes
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)
    
    # Check if categorical variables are encoded
    assert X.shape[1] == 12  # Original numerical + encoded categorical features
    
    # Check if target is encoded properly
    assert set(np.unique(y)) == {0, 1}

def test_split_data(sample_data):
    """Test data splitting functionality."""
    processor = DataProcessor()
    X, y = processor.preprocess_data(sample_data, 'Exited')
    
    # Test without validation split
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
    assert len(X_train) == int(0.8 * len(X))
    assert len(X_test) == int(0.2 * len(X))
    
    # Test with validation split
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X, y, test_size=0.2, validation_split=0.1
    )
    assert len(X_train) == int(0.7 * len(X))
    assert len(X_val) == int(0.1 * len(X))
    assert len(X_test) == int(0.2 * len(X))

def test_load_data_file_not_found():
    """Test proper error handling for missing files."""
    processor = DataProcessor()
    with pytest.raises(FileNotFoundError):
        processor.load_data('nonexistent_file.csv')

def test_get_preprocessed_data(sample_data, tmp_path):
    """Test complete preprocessing pipeline."""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    processor = DataProcessor()
    X_train, X_val, X_test, y_train, y_val, y_test = processor.get_preprocessed_data(
        data_path,
        'Exited',
        test_size=0.2,
        validation_split=0.1
    )
    
    # Check that all splits exist and have correct shapes
    assert X_train is not None
    assert X_val is not None
    assert X_test is not None
    assert y_train is not None
    assert y_val is not None
    assert y_test is not None
    
    # Check that splits have consistent feature dimensions
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]