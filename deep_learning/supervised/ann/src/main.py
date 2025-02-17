"""
Main Application Module for Bank Customer Churn Prediction
This is the entry point of the application that:
1. Orchestrates the data processing pipeline
2. Initializes and trains the neural network model
3. Performs model evaluation
4. Provides example predictions for new customers
"""

from data_processor import BankDataProcessor
from model_builder import BankChurnModel
from evaluation import ModelEvaluator
from sklearn.model_selection import KFold
import numpy as np

def main():
    # Initialize data processor
    processor = BankDataProcessor()
    
    # Load and preprocess data
    X, y = processor.load_data('Churn_Modelling.csv')
    X_train, X_test, y_train, y_test = processor.preprocess_data(X, y)
    
    # Initialize model
    model = BankChurnModel(input_dim=X_train.shape[1])
    
    # Train model
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    evaluator.plot_training_history(history)
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_pred_proba)
    
    # Example prediction
    sample_customer = [
        600,    # Credit Score
        'France',# Geography
        'Male',  # Gender
        40,     # Age
        3,      # Tenure
        60000,  # Balance
        2,      # Number of Products
        1,      # Has Credit Card
        1,      # Is Active Member
        50000   # Estimated Salary
    ]
    
    prediction = model.predict(processor.prepare_single_prediction(sample_customer))
    print(f"Probability of customer churning: {prediction[0][0]:.2%}")

if __name__ == "__main__":
    main()
