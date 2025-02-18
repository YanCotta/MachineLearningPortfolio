import os
import numpy as np
from src.data_processing.data_loader import DataLoader
from src.models.som_detector import SOMFraudDetector
from src.models.ann_classifier import ANNClassifier
from src.visualization.visualizer import FraudVisualization
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    try:
        # 1. Load and preprocess data
        data_loader = DataLoader('data/Credit_Card_Applications.csv')
        X_scaled, y, X_original = data_loader.load_and_preprocess()
        
        # 2. Unsupervised Learning Phase (SOM)
        som_detector = SOMFraudDetector(
            x=10, y=10,
            input_len=X_scaled.shape[1],
            sigma=1.0,
            learning_rate=0.5
        )
        som_detector.train(X_scaled)
        
        # Get potential fraud cases from SOM
        fraud_indices = som_detector.detect_anomalies(X_scaled)
        X_frauds = X_original[fraud_indices]
        
        # 3. Supervised Learning Phase (ANN)
        # Use SOM results as training data for ANN
        y_som = np.zeros(len(X_original))
        y_som[fraud_indices] = 1  # Mark SOM-detected frauds
        
        ann_classifier = ANNClassifier(
            input_dim=X_scaled.shape[1],
            hidden_layers=[6, 6],
            dropout_rate=0.3
        )
        
        # Train ANN with SOM-detected patterns
        ann_classifier.train(X_scaled, y_som)
        
        # 4. Final Predictions & Visualization
        final_predictions = ann_classifier.predict(X_scaled)
        
        # Visualize results
        visualizer = FraudVisualization(output_dir='output')
        visualizer.plot_som_heatmap(som_detector, X_scaled, y)
        visualizer.plot_fraud_distribution(final_predictions, X_original)
        visualizer.plot_feature_importance(ann_classifier, data_loader.feature_names)
        
        logger.info(f"Analysis complete. Detected {sum(final_predictions)} potential fraud cases.")
        return final_predictions, X_frauds
        
    except Exception as e:
        logger.error(f"Error in fraud detection pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()