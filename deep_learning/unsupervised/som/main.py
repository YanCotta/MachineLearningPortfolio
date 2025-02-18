import os
from src.dataset.data_loader import CreditCardDataLoader
from src.model.som_model import CreditCardSOM
from src.visualization.visualize import (
    plot_som_heatmap,
    plot_anomaly_distribution,
    plot_feature_importance
)

def main():
    # Initialize data loader
    data_path = os.path.join(os.path.dirname(__file__), 'dataset', 'Credit_Card_Applications.csv')
    loader = CreditCardDataLoader(data_path)
    X_scaled, y, X_original = loader.load_data()

    # Initialize and train SOM
    som = CreditCardSOM(x=10, y=10, input_len=15)
    som.train(X_scaled)

    # Get winning nodes and distance map
    winners = som.get_winning_nodes(X_scaled)
    distance_map = som.get_feature_map()

    # Plot results
    plot_som_heatmap(distance_map, winners, y, "Credit Card Applications SOM")
    
    # Find anomalies
    anomaly_scores = som.get_anomaly_scores(X_scaled)
    anomalies = som.find_anomalies(X_scaled)
    
    # Plot anomaly distribution
    threshold = som.get_anomaly_scores(X_scaled)[anomalies].min()
    plot_anomaly_distribution(anomaly_scores, threshold)
    
    # Plot feature importance
    feature_names = ['CustomerID', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 
                    'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14']
    plot_feature_importance(som, feature_names)

    # Print anomaly detection results
    anomalous_applications = X_original[anomalies]
    print(f"\nDetected {len(anomalous_applications)} potential fraudulent applications:")
    print(anomalous_applications)

if __name__ == "__main__":
    main()