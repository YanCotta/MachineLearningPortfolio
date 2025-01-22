# Multiple Linear Regression - Fundamental regression technique
# Models linear relationship between multiple independent variables and one dependent variable

# Core data science and analysis libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load and structure dataset:
# Assumes linear relationship between features and target
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Partition data into training and testing sets
# Standard 80-20 split for sufficient training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train multiple linear regression model
# Assumptions: linearity, independence, homoscedasticity, normality
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Model evaluation and predictions
# Compare predicted vs actual values
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)