# K-Means Clustering
# This script demonstrates customer segmentation using the K-means clustering algorithm.
# K-means is an iterative algorithm that partitions n observations into k clusters where
# each observation belongs to the cluster with the nearest mean (cluster centroid).

# Importing the required libraries
# numpy: for numerical operations
# matplotlib: for data visualization
# pandas: for data manipulation and analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading and preprocessing the dataset
# We select columns 3 (Annual Income) and 4 (Spending Score) as our features
# Other columns are excluded as they're either identifiers or not relevant for this analysis
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Determining the optimal number of clusters using the elbow method
# The elbow method runs k-means clustering for a range of k values (1-10)
# and plots the Within-Cluster Sum of Squares (WCSS) against k
# The "elbow" of the plot indicates the optimal k value
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ gives us the WCSS value
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model
# n_clusters = 5: chosen based on elbow method
# init = 'k-means++': uses smart initialization for better convergence
# random_state = 42: ensures reproducibility of results
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
# Each cluster is assigned a different color
# Centroids are marked in yellow
# The scatter plot helps identify distinct customer segments based on
# their annual income and spending score
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()