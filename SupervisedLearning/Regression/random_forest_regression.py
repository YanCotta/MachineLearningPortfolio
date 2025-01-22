# Random Forest Regression - Ensemble learning method using multiple decision trees
# Excellent for handling complex datasets with non-linear relationships

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset preparation:
# Ensure your CSV file has features in columns and target in the last column
dataset = pd.read_csv('Data.csv')  # Replace with your dataset filename
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data: 80% training, 20% testing
# Maintain data distribution with random_state
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Configure and train Random Forest model
# n_estimators: number of trees in the forest (increase for better accuracy)
# random_state: ensures reproducible results
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

# Generate predictions on test data
# Evaluate model accuracy by comparing predictions with actual values
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)