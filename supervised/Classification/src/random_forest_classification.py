# Random Forest Classification
# An ensemble learning method using multiple decision trees
# Best used for: Complex classification tasks with large datasets
# Advantages: High accuracy, handles overfitting, feature importance
# Limitations: Black box model, computationally intensive

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Random Forest Classifier
# Important Parameters:
# - n_estimators: Number of trees (default=100)
# - criterion: 'gini' or 'entropy'
# - max_depth: Maximum depth of trees
# - min_samples_split: Minimum samples required to split
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=10,
    criterion='entropy',
    random_state=0
    # max_depth=None,  # Uncomment to control tree depth
    # min_samples_split=2  # Uncomment to adjust splitting
)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Optional: Feature Importance Analysis
importances = classifier.feature_importances_
print("\nFeature Importances:")
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")