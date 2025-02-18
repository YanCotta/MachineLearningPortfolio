import os
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_processing.data_loader import DataLoader
from src.models.som_detector import SOMFraudDetector
from src.models.ann_classifier import ANNClassifier
from src.visualization.visualizer import FraudVisualization
from src.utils.logger import setup_logger
import tensorflow as tf

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hybrid Fraud Detection System combining SOM and ANN'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/Credit_Card_Applications.csv',
        help='Path to the credit card applications dataset'
    )
    
    # SOM arguments
    parser.add_argument(
        '--som-grid-size',
        type=int,
        default=10,
        help='Size of the SOM grid (will be used for both dimensions)'
    )
    parser.add_argument(
        '--som-sigma',
        type=float,
        default=1.0,
        help='Initial neighborhood radius for SOM'
    )
    parser.add_argument(
        '--som-learning-rate',
        type=float,
        default=0.5,
        help='Initial learning rate for SOM'
    )
    
    # ANN arguments
    parser.add_argument(
        '--ann-layers',
        type=str,
        default='auto',
        help='Comma-separated list of hidden layer sizes (e.g., "64,32") or "auto"'
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.3,
        help='Dropout rate for ANN layers'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs for ANN'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save visualizations and results'
    )
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    try:
        # Enable memory growth for GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Running with GPU acceleration: {len(gpus)} GPU(s) available")
        else:
            logger.info("Running on CPU")

        # Ensure correct data path resolution
        base_path = Path(__file__).parent
        data_path = base_path / args.data_path

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        # 1. Load and preprocess data
        data_loader = DataLoader(str(data_path))
        X_scaled, y, X_original = data_loader.load_and_preprocess()
        
        if X_scaled.shape[0] == 0:
            raise ValueError("No data loaded from dataset")

        # 2. Unsupervised Learning Phase (SOM)
        input_dim = X_scaled.shape[1]
        som_detector = SOMFraudDetector(
            x=args.som_grid_size,
            y=args.som_grid_size,
            input_len=input_dim,
            sigma=args.som_sigma,
            learning_rate=args.som_learning_rate
        )
        som_detector.train(X_scaled)
        
        # Get potential fraud cases from SOM
        fraud_indices = som_detector.detect_anomalies(X_scaled)
        if len(fraud_indices) == 0:
            logger.warning("SOM detected no anomalies. Adjusting threshold...")
            fraud_indices = som_detector.detect_anomalies(X_scaled, threshold=0.8)
        
        X_frauds = X_original[fraud_indices]
        
        # 3. Supervised Learning Phase (ANN)
        # Use SOM results as training data for ANN
        y_som = np.zeros(len(X_original))
        y_som[fraud_indices] = 1  # Mark SOM-detected frauds
        
        # Verify we have both positive and negative samples
        if not (np.any(y_som == 0) and np.any(y_som == 1)):
            raise ValueError("Training data must contain both fraud and non-fraud cases")
        
        # Configure ANN architecture
        if args.ann_layers == 'auto':
            hidden_layers = [input_dim, input_dim//2]
        else:
            hidden_layers = [int(x) for x in args.ann_layers.split(',')]
        
        ann_classifier = ANNClassifier(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=args.dropout_rate
        )
        
        # Train ANN with SOM-detected patterns
        history = ann_classifier.train(
            X_scaled, y_som,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=32
        )
        
        # 4. Final Predictions & Visualization
        final_predictions = ann_classifier.predict(X_scaled)
        
        # Create output directory if it doesn't exist
        output_dir = base_path / args.output_dir
        output_dir.mkdir(exist_ok=True)
        
        # Initialize visualizer
        visualizer = FraudVisualization(str(output_dir))
        
        # Generate all visualizations
        visualizer.plot_som_heatmap(som_detector, X_scaled, y)
        visualizer.plot_fraud_distribution(
            final_predictions,
            X_original,
            data_loader.feature_names
        )
        visualizer.plot_feature_importance(
            ann_classifier, 
            data_loader.feature_names,
            X_scaled
        )
        visualizer.plot_model_metrics(y, final_predictions)
        
        n_frauds = np.sum(final_predictions)
        logger.info(f"Analysis complete. Detected {n_frauds} potential fraud cases "
                f"({n_frauds/len(final_predictions)*100:.2f}% of total)")
        
        return final_predictions, X_frauds
        
    except Exception as e:
        logger.error(f"Error in fraud detection pipeline: {str(e)}")
        raise
    finally:
        # Cleanup GPU memory and matplotlib resources
        tf.keras.backend.clear_session()
        plt.close('all')

if __name__ == "__main__":
    main()