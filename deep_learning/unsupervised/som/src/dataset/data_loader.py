import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class CreditCardDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """Load and preprocess the credit card application dataset."""
        dataset = pd.read_csv(self.filepath)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, X
    
    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale."""
        return self.scaler.inverse_transform(X_scaled)