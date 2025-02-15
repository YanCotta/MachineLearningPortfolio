# Artificial Neural Network for Customer Churn Prediction

## Overview
This project implements a sophisticated artificial neural network (ANN) for predicting customer churn in a banking context. Built from scratch using Python, it demonstrates advanced deep learning concepts while providing a practical business application.

## Features
- Modular implementation of neural network components
- Support for multiple hidden layers with configurable architectures
- Various activation functions (ReLU, Sigmoid, Tanh)
- Advanced optimization techniques:
  - Mini-batch gradient descent
  - Adaptive learning rates
  - Momentum-based optimization
- Comprehensive loss functions:
  - Binary cross-entropy
  - Mean squared error (MSE)
  - Categorical cross-entropy
- Robust model evaluation metrics
- Data preprocessing utilities
- Model persistence capabilities

## Project Structure
```
ann/
├── __init__.py
├── model_builder.py      # Neural network architecture definition
├── data_processor.py     # Data preprocessing and manipulation
├── main.py              # Main training and prediction pipeline
├── evaluation.py        # Model evaluation metrics and visualization
└── Churn_Modelling.csv  # Banking customer dataset
```

## Dataset
The project uses the Bank Customer Churn Prediction dataset (`Churn_Modelling.csv`) containing the following features:
- Customer demographic information (age, gender, geography)
- Banking relationship metrics (tenure, credit score, balance)
- Product usage indicators (number of products, credit card status)
- Target variable: Customer churn status (0 = retained, 1 = churned)

## Installation
```bash
git clone <repository-url>
cd MachineLearningPortfolio/deep_learning/ann
pip install -r requirements.txt
```

## Usage
### Basic Implementation
```python
from data_processor import DataProcessor
from model_builder import NeuralNetwork

# Load and preprocess data
processor = DataProcessor('Churn_Modelling.csv')
X_train, X_test, y_train, y_test = processor.prepare_data()

# Create and train model
model = NeuralNetwork([
    {'units': 6, 'activation': 'relu', 'input_dim': X_train.shape[1]},
    {'units': 6, 'activation': 'relu'},
    {'units': 1, 'activation': 'sigmoid'}
])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate performance
accuracy = model.evaluate(X_test, y_test)
```

### Advanced Configuration
```python
# Custom training configuration
model.fit(
    X_train, 
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    learning_rate=0.001,
    momentum=0.9,
    early_stopping=True,
    patience=10
)
```

## Model Architecture
The neural network implementation supports:
- Flexible layer configuration
- Multiple activation functions
- Dropout regularization
- Batch normalization
- Weight initialization techniques

## Performance Metrics
The model evaluation includes:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citations
If you use this implementation in your research, please cite:
```
@misc{ann_churn_prediction,
  author = Yan Cotta,
  title = {Neural Network Implementation for Customer Churn Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YanCotta/MachineLearningPortfolio}
}
