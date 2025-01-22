# Support Vector Regression (SVR) - A powerful regression model that works well with non-linear data
# This template implements SVR with RBF kernel for complex regression tasks

# Import essential data manipulation and visualization libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load and prepare the dataset:
# X contains all independent features, y contains the target variable
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Create training and validation sets with 80-20 split
# random_state ensures reproducibility of results
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Standardize features and target variable
# SVR is sensitive to feature scales, so standardization is crucial
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Initialize and train SVR model
# kernel='rbf' uses Radial Basis Function for non-linear relationships
# Adjust C, epsilon, and gamma parameters for fine-tuning
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Make predictions and inverse transform to original scale
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Calculate R-squared score to evaluate model performance
# R-squared closer to 1 indicates better fit
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)