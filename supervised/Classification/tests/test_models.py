import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestClassificationModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        # Load sample data
        cls.data = pd.read_csv('../data/Social_Network_Ads.csv')
        X = cls.data.iloc[:, :-1].values
        y = cls.data.iloc[:, -1].values
        
        # Split and scale data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        sc = StandardScaler()
        cls.X_train = sc.fit_transform(cls.X_train)
        cls.X_test = sc.transform(cls.X_test)

    def test_data_loading(self):
        """Test if data is loaded correctly"""
        self.assertIsNotNone(self.data)
        self.assertEqual(len(self.data.columns), 3)
        self.assertTrue('Purchased' in self.data.columns)

    def test_data_preprocessing(self):
        """Test data preprocessing steps"""
        self.assertEqual(self.X_train.shape[1], 2)
        self.assertTrue(np.isclose(self.X_train.mean(), 0, atol=1e-7))
        self.assertTrue(np.isclose(self.X_train.std(), 1, atol=1e-7))

if __name__ == '__main__':
    unittest.main()
