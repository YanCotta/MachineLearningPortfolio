"""
Main script for training and evaluating the CNN model.
"""

import logging
import argparse
from pathlib import Path
from src.data_processor.data_loader import DataLoader
from src.model.cnn_model import CNNModel
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate CNN model')
    parser.add_argument('--train-dir', type=str, required=True,
                    help='Directory containing training data')
    parser.add_argument('--test-dir', type=str, required=True,
                    help='Directory containing test data')
    parser.add_argument('--epochs', type=int, default=25,
                    help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for training')
    parser.add_argument('--img-size', type=int, nargs=2, default=[64, 64],
                    help='Image dimensions (height width)')
    parser.add_argument('--experiment-name', type=str,
                    help='Name for this training run')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()

    try:
        # Initialize components
        data_loader = DataLoader(
            img_size=tuple(args.img_size),
            batch_size=args.batch_size
        )
        model = CNNModel(img_size=tuple(args.img_size))
        trainer = ModelTrainer(
            model,
            experiment_name=args.experiment_name
        )
        evaluator = ModelEvaluator()

        # Load and prepare data
        logger.info("Loading datasets...")
        train_data, validation_data, test_data = data_loader.load_data(
            args.train_dir,
            args.test_dir
        )

        # Build and train model
        logger.info("Building model...")
        model.build()

        logger.info("Starting training...")
        history = trainer.train(
            train_data,
            validation_data,
            epochs=args.epochs
        )

        # Evaluate model
        logger.info("Evaluating model...")
        evaluator.set_training_history(history)
        
        # Create output directory for plots
        output_dir = Path("outputs") / (args.experiment_name or "default")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save evaluation plots
        evaluator.plot_training_history(
            save_path=str(output_dir / "training_history.png")
        )

        eval_results = evaluator.evaluate_model(model.model, test_data)
        evaluator.plot_confusion_matrix(
            eval_results['confusion_matrix'],
            save_path=str(output_dir / "confusion_matrix.png")
        )

        evaluator.visualize_predictions(
            model.model,
            test_data,
            save_path=str(output_dir / "sample_predictions.png")
        )

        logger.info(f"Evaluation results and visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()