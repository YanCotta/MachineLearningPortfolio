import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
from datetime import datetime
from data_processor import DataProcessor
from model_builder import NeuralNetworkBuilder
from evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_callbacks(checkpoint_dir: str) -> list:
    """
    Set up training callbacks for model optimization.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        
    Returns:
        list: List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks

def main():
    """Main execution function."""
    try:
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join('runs', timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize data processor
        logger.info("Initializing data preprocessing")
        data_processor = DataProcessor()
        
        # Load and preprocess data
        dataset_path = os.path.join('dataset', 'Churn_Modelling.csv')
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.get_preprocessed_data(
            filepath=dataset_path,
            target_column='Exited',
            validation_split=0.1
        )
        
        # Initialize model builder
        logger.info("Building neural network model")
        model_builder = NeuralNetworkBuilder(input_dim=X_train.shape[1])
        
        # Build model with advanced architecture
        model = model_builder.build_model(
            hidden_layers=[32, 16, 8],
            dropout_rates=[0.3, 0.2, 0.1],
            use_batch_norm=True,
            kernel_regularizer={'l1': 1e-5, 'l2': 1e-4}
        )
        
        # Compile model
        model_builder.compile_model(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        # Setup callbacks
        callbacks = setup_callbacks(os.path.join(run_dir, 'checkpoints'))
        
        # Train model
        logger.info("Starting model training")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating model performance")
        evaluator = ModelEvaluator(model, save_dir=os.path.join(run_dir, 'evaluation'))
        metrics = evaluator.evaluate_model(X_test, y_test, history)
        
        # Analyze predictions
        feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                        'Geography_France', 'Geography_Germany', 'Geography_Spain',
                        'Gender_Female', 'Gender_Male']
        prediction_analysis = evaluator.analyze_predictions(X_test, y_test, feature_names)
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
