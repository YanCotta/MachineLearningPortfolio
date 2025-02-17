# 🧠 Customer Churn Prediction using Artificial Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/YanCotta/MachineLearningPortfolio/graphs/commit-activity)

> A sophisticated deep learning solution for predicting customer churn in banking, implemented from scratch using Python.

## 🎯 Key Features

- **Modular Architecture**
  - Component-based neural network implementation
  - Configurable multi-layer support
  - Flexible model customization

- **Advanced Capabilities**
  - Multiple activation functions (ReLU, Sigmoid, Tanh)
  - Sophisticated optimization techniques
  - Comprehensive loss function selection
  - Robust evaluation metrics suite

- **Production-Ready**
  - Efficient data preprocessing pipeline
  - Model persistence and versioning
  - Performance optimization features

## 🏗️ Project Structure
```
ann/
├── __init__.py
├── model_builder.py      # Neural network architecture definition
├── data_processor.py     # Data preprocessing and manipulation
├── main.py              # Main training and prediction pipeline
├── evaluation.py        # Model evaluation metrics and visualization
└── Churn_Modelling.csv  # Banking customer dataset
```

## 📊 Dataset Overview

The model is trained on a comprehensive banking dataset (`Churn_Modelling.csv`) containing:

| Category | Features |
|----------|----------|
| Demographics | Age, Gender, Geography |
| Banking Metrics | Tenure, Credit Score, Balance |
| Product Usage | Number of Products, Credit Card Status |
| Target Variable | Churn Status (0: Retained, 1: Churned) |

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd MachineLearningPortfolio/deep_learning/ann
pip install -r requirements.txt
```

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

## 🎛️ Model Architecture

### Supported Features
- **Layer Configuration**
  - Dynamic layer depth and width
  - Customizable activation functions
  - Dropout regularization
  - Batch normalization

- **Optimization**
  - Mini-batch gradient descent
  - Adaptive learning rates
  - Momentum optimization

## 📈 Performance Metrics

Our comprehensive evaluation suite includes:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall prediction accuracy |
| Precision | True positive prediction accuracy |
| Recall | Positive case detection rate |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Classification performance at various thresholds |
| Confusion Matrix | Detailed prediction breakdown |

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

```bibtex
@misc{ann_churn_prediction,
  author = {Yan Cotta},
  title = {Neural Network Implementation for Customer Churn Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YanCotta/MachineLearningPortfolio}
}
```

---
<p align="center">
  <i>Built with ❤️ by <a href="https://github.com/YanCotta">Yan Cotta</a></i>
</p>
