"""Simple Linear Regression implementation using scikit-learn.

This module provides a straightforward implementation of simple linear regression
for predicting a continuous target variable based on a single feature.
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

class SimpleLinearRegression:
    """Simple Linear Regression model implementation.
    
    Attributes:
        regressor: sklearn.linear_model.LinearRegression instance
    """
    
    def __init__(self):
        """Initialize the linear regression model."""
        self.regressor = LinearRegression()

    def fit(self, X, y):
        """Train the model using the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, 1)
            y: Target vector of shape (n_samples,)
        """
        self.regressor.fit(X, y)
        return self

# Train the Simple Linear Regression model on the Training set
model = SimpleLinearRegression()
model.fit(X_train, y_train)

# Predict the Test set results
y_pred = model.regressor.predict(X_test)

# Visualize the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, model.regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, model.regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()