import os
import logging
from datetime import datetime
from src.dataset.data_loader import CreditCardDataLoader
from src.model.som_model import CreditCardSOM
from src.visualization.visualize import (
    plot_som_heatmap,
    plot_anomaly_distribution,
    plot_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_output_dir():
    """Create output directory for results and visualizations."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 
                             datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    try:
        # Create output directory
        output_dir = setup_output_dir()
        logger.info(f"Output directory created at: {output_dir}")

        # Initialize data loader
        data_path = os.path.join(os.path.dirname(__file__), 'dataset', 'Credit_Card_Applications.csv')
        loader = CreditCardDataLoader(data_path)
        logger.info("Loading and preprocessing data...")
        X_scaled, y, X_original = loader.load_data()
        logger.info(f"Loaded dataset with {X_scaled.shape[0]} samples and {X_scaled.shape[1]} features")

        # Initialize and train SOM
        logger.info("Initializing and training SOM...")
        som = CreditCardSOM(x=10, y=10, input_len=X_scaled.shape[1])
        som.train(X_scaled)
        logger.info("SOM training completed")

        # Generate and save visualizations
        logger.info("Generating visualizations...")
        
        # Get winning nodes and distance map
        winners = som.get_winning_nodes(X_scaled)
        distance_map = som.get_feature_map()

        # Plot and save SOM heatmap
        plot_som_heatmap(
            distance_map, winners, y, 
            "Credit Card Applications SOM",
            save_path=os.path.join(output_dir, 'som_heatmap.png')
        )
        
        # Find and analyze anomalies
        anomaly_scores = som.get_anomaly_scores(X_scaled)
        anomalies = som.find_anomalies(X_scaled)
        
        # Plot and save anomaly distribution
        threshold = som.get_anomaly_scores(X_scaled)[anomalies].min()
        plot_anomaly_distribution(
            anomaly_scores, threshold,
            save_path=os.path.join(output_dir, 'anomaly_distribution.png')
        )
        
        # Plot and save feature importance
        feature_names = ['CustomerID', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 
                        'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14']
        plot_feature_importance(
            som, feature_names,
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )

        # Save analysis results
        anomalous_applications = X_original[anomalies]
        logger.info(f"\nDetected {len(anomalous_applications)} potential fraudulent applications")
        
        # Save results to file
        results_path = os.path.join(output_dir, 'fraud_detection_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Applications Analyzed: {len(X_original)}\n")
            f.write(f"Potential Fraudulent Applications Detected: {len(anomalous_applications)}\n\n")
            f.write("Detailed Results:\n")
            f.write(str(anomalous_applications))
            
        logger.info(f"Results saved to: {results_path}")
        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()