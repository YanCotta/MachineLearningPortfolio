# Polynomial Regression - Extends linear regression for non-linear relationships
# Transforms features into polynomial features for complex pattern recognition

# Required libraries for data handling and analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data preprocessing steps:
# X: feature matrix, y: target vector
dataset = pd.read_csv('Data.csv')  # Replace with your dataset filename
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Create train-test split for model validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Transform features into polynomial features
# degree parameter controls complexity (higher degree = more complex model)
# Warning: High degrees may lead to overfitting
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)

# Fit model and make predictions
# Uses transformed polynomial features for better curve fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# Predicting the Test set results
y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)