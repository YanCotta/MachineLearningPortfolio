# Decision Tree Regression - Non-linear regression using binary tree structure
# Splits data into branches based on feature values to make predictions

# Standard numerical and data processing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import and structure dataset:
# Features should be meaningful for splitting decisions
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Create training and validation sets
# Validation ensures tree doesn't overfit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Configure and train decision tree
# Can capture non-linear patterns without feature scaling
# Consider max_depth and min_samples_split parameters to prevent overfitting
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Generate and evaluate predictions
# Tree structure allows for interpretable decision paths
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)