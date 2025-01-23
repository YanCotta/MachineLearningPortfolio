import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Add parent directory to path to import model files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simple_linear_regression import LinearRegression
from src.polynomial_regression import PolynomialFeatures
from src.random_forest_regression import RandomForestRegressor

class TestModels(unittest.TestCase):
    def test_data_shapes(self):
        df_50 = pd.read_csv('../data/50_Startups.csv')
        df_positions = pd.read_csv('../data/Position_Salaries.csv')
        df_salary = pd.read_csv('../data/Salary_Data.csv')
        self.assertTrue(df_50.shape[0] > 0)
        self.assertTrue(df_positions.shape[0] > 0)
        self.assertTrue(df_salary.shape[0] > 0)

class TestRegressionModels(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        self.X, self.y = make_regression(n_samples=100, n_features=1, noise=0.1)
        self.y = self.y.reshape(-1, 1)

    def test_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_polynomial_features(self):
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.X)
        self.assertEqual(X_poly.shape[1], 3)  # Original + x^1 + x^2

    def test_random_forest(self):
        model = RandomForestRegressor(n_estimators=10)
        model.fit(self.X, self.y.ravel())
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

if __name__ == '__main__':
    unittest.main()