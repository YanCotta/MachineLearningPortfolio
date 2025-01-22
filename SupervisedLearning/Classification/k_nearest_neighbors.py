# K-Nearest Neighbors (K-NN)
# Instance-based learning algorithm that classifies based on closest training examples
# Best used for: Small to medium datasets with low dimensionality
# Advantages: Simple, no assumptions about data, works with multiclass
# Limitations: Computationally expensive, sensitive to irrelevant features

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

# Feature Scaling is crucial for K-NN
# Distance calculations are directly affected by feature scales
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the K-NN model on the Training set
# Parameters explained:
# - n_neighbors: Number of neighbors (k)
# - metric: Distance measure ('minkowski' with p=2 is Euclidean)
# - weights: 'uniform' or 'distance' based weighting
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2,  # p=2 for Euclidean distance
    # weights='uniform'  # Uncomment to adjust neighbor weights
)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Optional: Find optimal k value
# from sklearn.model_selection import cross_val_score
# k_range = range(1, 31)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=5)
#     k_scores.append(scores.mean())