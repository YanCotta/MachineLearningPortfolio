import matplotlib.pyplot as plt
import numpy as np

def plot_som_heatmap(distance_map, winners, y=None, title="SOM Heatmap", save_path=None):
    """Plot SOM heatmap with data points.
    
    Args:
        distance_map: Distance map from trained SOM
        winners: Winning nodes for each data point
        y: Optional class labels for coloring
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.pcolor(distance_map.T, cmap='bone_r')
    plt.colorbar()
    
    if y is not None:
        markers = ['o', 's']
        colors = ['r', 'g']
        for i, w in enumerate(winners):
            plt.plot(w[0] + 0.5,
                    w[1] + 0.5,
                    markers[y[i]],
                    markeredgecolor=colors[y[i]],
                    markerfacecolor='None',
                    markersize=10,
                    markeredgewidth=2)
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt.gcf()

def plot_anomaly_distribution(anomaly_scores, threshold=None, save_path=None):
    """Plot distribution of anomaly scores.
    
    Args:
        anomaly_scores: List of anomaly scores
        threshold: Optional threshold value to mark
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, bins=50, density=True, alpha=0.7)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.2f}')
        plt.legend()
    
    plt.title("Distribution of Anomaly Scores")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt.gcf()

def plot_feature_importance(som, feature_names, save_path=None):
    """Plot feature importance based on SOM weights.
    
    Args:
        som: Trained SOM model
        feature_names: List of feature names
        save_path: Optional path to save the plot
    """
    if not hasattr(som, 'som'):
        raise ValueError("The provided SOM model is invalid")
        
    weights = som.som.get_weights()
    importance = np.std(weights, axis=(0,1))
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), feature_names, rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Weight Variance")
    plt.title("Feature Importance Based on SOM Weights")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt.gcf()