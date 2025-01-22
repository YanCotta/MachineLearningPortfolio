# Logistic Regression

# A fundamental classification algorithm that predicts binary outcomes (0/1, Yes/No, True/False)
# Best used for: Binary classification problems with linearly separable classes
# Advantages: Simple, interpretable, requires less computational power
# Limitations: Assumes linear relationship, may underperform with non-linear data

# Importing the libraries
import numpy as np          # For numerical operations and array manipulation
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd         # For data manipulation and analysis

# Importing the dataset
# Replace 'ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv' with your dataset filename
# Dataset should be preprocessed with:
# - No missing values
# - Features in numerical format (encode categorical variables)
# - Binary target variable
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values  # Features: Select all columns except the last one
y = dataset.iloc[:, -1].values   # Target: Select only the last column

# Splitting the dataset into the Training set and Test set
# test_size=0.25 means 75% of data for training, 25% for testing
# random_state ensures reproducibility of results
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
# Standardizes features by removing the mean and scaling to unit variance
# Important for logistic regression to:
# - Prevent features with larger scales from dominating the model
# - Speed up gradient descent convergence
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit to training data and transform it
X_test = sc.transform(X_test)        # Transform test data using training data parameters

# Training the Logistic Regression model on the Training set
# Parameters:
# - random_state: Ensures reproducibility
# Consider tuning these parameters:
# - C: Inverse of regularization strength (default=1.0)
# - penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
# - solver: Algorithm to use ('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Making predictions and evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)

# Confusion Matrix shows:
# - True Negatives (top-left)
# - False Positives (top-right)
# - False Negatives (bottom-left)
# - True Positives (bottom-right)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Accuracy = (True Positives + True Negatives) / Total Predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.2f}")

# Optional: Add model evaluation metrics
# from sklearn.metrics import classification_report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))