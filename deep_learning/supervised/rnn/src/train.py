import argparse
import logging
import os
from datetime import datetime
from data_processor import DataProcessor
from model import StockPredictor
import matplotlib.pyplot as plt
import numpy as np

def setup_logging():
    """Configure logging settings"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_results(actual, predicted, title, save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train stock price prediction model')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length for LSTM')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize data processor
        data_processor = DataProcessor(args.train_data, args.test_data)
        X_train, y_train, X_test, y_test = data_processor.load_and_preprocess_data()
        
        # Initialize and train model
        model = StockPredictor(sequence_length=args.sequence_length)
        history = model.train(
            X_train, 
            y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Inverse transform predictions
        train_predictions = data_processor.inverse_transform(train_predictions)
        test_predictions = data_processor.inverse_transform(test_predictions)
        y_train_actual = data_processor.inverse_transform(y_train)
        y_test_actual = data_processor.inverse_transform(y_test)
        
        # Plot results
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training results
        plot_results(
            y_train_actual,
            train_predictions,
            'Stock Price Prediction - Training Data',
            os.path.join(plots_dir, 'training_results.png')
        )
        
        # Plot test results
        plot_results(
            y_test_actual,
            test_predictions,
            'Stock Price Prediction - Test Data',
            os.path.join(plots_dir, 'test_results.png')
        )
        
        # Plot training history
        plt.figure(figsize=(15, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'training_history.png'))
        plt.close()
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()