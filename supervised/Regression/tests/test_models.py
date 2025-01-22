
import unittest
import pandas as pd
import numpy as np

class TestModels(unittest.TestCase):
    def test_data_shapes(self):
        df_50 = pd.read_csv('../data/50_Startups.csv')
        df_positions = pd.read_csv('../data/Position_Salaries.csv')
        df_salary = pd.read_csv('../data/Salary_Data.csv')
        self.assertTrue(df_50.shape[0] > 0)
        self.assertTrue(df_positions.shape[0] > 0)
        self.assertTrue(df_salary.shape[0] > 0)

if __name__ == '__main__':
    unittest.main()