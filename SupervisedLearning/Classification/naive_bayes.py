# Naive Bayes Classifier
# A probabilistic classifier based on Bayes' theorem
# Best used for: Text classification, spam filtering, sentiment analysis
# Advantages: Fast, efficient, works well with high-dimensional data
# Limitations: Assumes feature independence (naive assumption)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset requirements:
# - Independent features (ideally)
# - Can handle both continuous and discrete data
# - Works well with both small and large datasets

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Note: Feature scaling is optional for Naive Bayes
# Including it here for consistency with other models

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model
# Using GaussianNB for continuous features
# Other variants:
# - MultinomialNB for discrete features
# - BernoulliNB for binary features
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB(
    # priors=None,  # Uncomment to set prior probabilities
    # var_smoothing=1e-9  # Uncomment to adjust variance calculation
)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

print("\nModel Information:")
print(f"Number of classes: {len(classifier.classes_)}")
print(f"Prior probabilities: {classifier.class_prior_}")