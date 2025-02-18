import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from data_processor import DataProcessor
from model import StockPredictor
from evaluation import ModelEvaluator
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train stock prediction model')
    parser.add_argument('--train-file', type=str, required=True,
                    help='Path to training data CSV')
    parser.add_argument('--test-file', type=str, required=True,
                    help='Path to test data CSV')
    parser.add_argument('--sequence-length', type=int, default=60,
                    help='Length of input sequences')
    parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                    help='Training batch size')
    parser.add_argument('--experiment-name', type=str,
                    help='Name for this training run')
    parser.add_argument('--use-features', action='store_true',
                    help='Use technical indicators as additional features')
    return parser.parse_args()

def setup_experiment_folder(experiment_name: str = None) -> Path:
    """Create and return experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = experiment_name or f"experiment_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    
    # Create necessary subdirectories
    for subdir in ['models', 'plots', 'logs', 'evaluation']:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def main():
    """Main execution function."""
    args = parse_args()

    try:
        # Set up experiment directory
        exp_dir = setup_experiment_folder(args.experiment_name)
        
        # Initialize components
        data_processor = DataProcessor()
        
        # Load and process training data
        logger.info("Loading and processing data...")
        train_df = data_processor.load_data(args.train_file)
        test_df = data_processor.load_data(args.test_file)
        
        # Add technical indicators if specified
        if args.use_features:
            logger.info("Adding technical indicators...")
            train_df = data_processor.add_technical_indicators(train_df)
            test_df = data_processor.add_technical_indicators(test_df)
            features = ['Close', 'MA7', 'MA20', 'RSI', 'MACD', 'BB_middle']
        else:
            features = ['Close']
        
        # Prepare sequences for training
        X_train, X_test, y_train, y_test = data_processor.prepare_data(
            train_df,
            sequence_length=args.sequence_length,
            train_split=0.8,
            features=features
        )
        
        # Initialize model
        logger.info("Building model...")
        n_features = len(features)
        model = StockPredictor(
            sequence_length=args.sequence_length,
            n_features=n_features,
            lstm_units=[100, 50, 50],
            dropout_rates=[0.3, 0.2, 0.2],
            bidirectional=True,
            use_batch_norm=True,
            save_path=str(exp_dir / 'models')
        )
        
        # Print model summary
        with open(exp_dir / 'logs' / 'model_summary.txt', 'w') as f:
            f.write(model.get_model_summary())
        
        # Train model
        logger.info("Starting training...")
        history = model.train(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Initialize evaluator
        evaluator = ModelEvaluator(output_dir=str(exp_dir / 'evaluation'))
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Inverse transform predictions
        y_train_actual = data_processor.inverse_transform(y_train)
        y_train_pred = data_processor.inverse_transform(y_pred_train)
        y_test_actual = data_processor.inverse_transform(y_test)
        y_test_pred = data_processor.inverse_transform(y_pred_test)
        
        # Plot results
        evaluator.plot_training_history(
            history.history,
            save_path='training_history.png'
        )
        
        evaluator.plot_predictions(
            y_test_actual,
            y_test_pred,
            title='Stock Price Prediction - Test Data',
            save_path='test_predictions.png'
        )
        
        # Analyze predictions
        analysis = evaluator.analyze_predictions(
            y_test_actual,
            y_test_pred,
            dates=test_df.index[-len(y_test_actual):]
        )
        
        # Generate and save report
        report = evaluator.generate_report(analysis)
        logger.info("\nEvaluation Report:\n" + report)
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()