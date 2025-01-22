# Support Vector Machine (SVM)
# A powerful classification algorithm that finds the optimal hyperplane to separate classes
# Best used for: Binary/multiclass classification with clear margins between classes
# Advantages: Effective in high-dimensional spaces, memory efficient
# Limitations: Not suitable for large datasets, sensitive to feature scaling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# Dataset requirements
# - Clean, normalized data
# - No missing values
# - Numeric features (encode categorical variables)
# - Labeled target variable
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling is crucial for SVM performance
# SVMs are sensitive to the scale of input features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
# Parameters explained:
# - kernel: 'linear' for linearly separable data
# - C: Regularization parameter (default=1.0)
#      Smaller C = Larger margin, more regularization
# - random_state: For reproducibility
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Evaluation metrics
y_pred = classifier.predict(X_test)

# Calculating and displaying the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculating and displaying the Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.2f}")

# Optional: Display classification report for more detailed metrics
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))