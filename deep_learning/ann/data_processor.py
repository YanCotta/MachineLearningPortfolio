import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class BankDataProcessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.onehot = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(sparse_output=False), [1])],
            remainder='passthrough'
        )
        
    def load_data(self, filepath):
        """Load and preprocess the bank data"""
        dataset = pd.read_csv(filepath)
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2):
        """Preprocess features and split data"""
        # Encode categorical variables
        X[:, 2] = self.label_encoder.fit_transform(X[:, 2])  # Gender
        X = self.onehot.fit_transform(X)  # Geography
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_single_prediction(self, data):
        """Prepare single instance for prediction"""
        data_array = np.array([data])
        encoded_data = self.onehot.transform(data_array)
        scaled_data = self.scaler.transform(encoded_data)
        return scaled_data
