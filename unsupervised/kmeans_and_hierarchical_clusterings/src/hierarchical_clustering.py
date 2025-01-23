# Hierarchical Clustering
# This script implements Agglomerative Hierarchical Clustering for customer segmentation.
# Unlike k-means, hierarchical clustering creates a tree-like hierarchy of clusters,
# allowing us to understand the nested structure of the data.

# Importing the required libraries
# numpy: for numerical operations
# matplotlib: for data visualization
# pandas: for data manipulation and analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading and preprocessing the dataset
# Similar to k-means, we focus on Annual Income and Spending Score
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Creating a dendrogram to find the optimal number of clusters
# The dendrogram shows the hierarchical relationship between points
# Vertical lines show the distance (dissimilarity) between clusters
# method='ward': minimizes variance within clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model
# n_clusters = 5: chosen based on dendrogram analysis
# affinity = 'euclidean': uses euclidean distance metric
# linkage = 'ward': minimizes variance within clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters
# Similar to k-means visualization, but without centroids
# Each color represents a distinct customer segment
# The plot helps identify natural groupings in customer behavior
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

def main():
    """Run Hierarchical clustering process."""
    pass