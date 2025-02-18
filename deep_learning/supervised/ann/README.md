# Advanced Neural Network Implementation

A production-grade implementation of an Artificial Neural Network (ANN) for binary classification, featuring comprehensive data preprocessing, modular architecture, and advanced evaluation metrics.

## 🌟 Features

- **Modular Architecture**
  - Separate modules for data processing, model building, and evaluation
  - Clean, maintainable, and well-documented code
  - Type hints and comprehensive error handling

- **Advanced Data Preprocessing**
  - Automated handling of categorical variables
  - Feature scaling and normalization
  - Flexible train/validation/test splitting
  - Comprehensive data validation

- **Sophisticated Model Architecture**
  - Configurable layer architecture
  - Dropout and BatchNormalization for regularization
  - L1/L2 regularization support
  - Learning rate scheduling
  - Early stopping and model checkpointing

- **Comprehensive Evaluation**
  - ROC curves and AUC metrics
  - Precision-Recall analysis
  - Confusion matrix visualization
  - Detailed prediction analysis
  - Training history visualization

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.8+
- Additional dependencies in `requirements.txt`

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd ann
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the model:
```bash
python src/main.py
```

## 🏗️ Project Structure

```
ann/
├── dataset/
│   └── Churn_Modelling.csv
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model_builder.py
│   ├── evaluation.py
│   └── main.py
├── tests/
│   └── ...
├── requirements.txt
└── README.md
```

## 📊 Model Architecture

The neural network implements a binary classification model with:
- Configurable hidden layers with ReLU activation
- Dropout layers for regularization
- Batch normalization for training stability
- Binary cross-entropy loss function
- Adam optimizer with learning rate scheduling

## 📈 Performance Metrics

The model evaluation includes:
- Classification accuracy
- ROC-AUC score
- Precision and recall metrics
- Confusion matrix
- Learning curves

## 🔧 Customization

You can customize the model architecture by modifying parameters in `main.py`:
```python
model = model_builder.build_model(
    hidden_layers=[32, 16, 8],
    dropout_rates=[0.3, 0.2, 0.1],
    use_batch_norm=True,
    kernel_regularizer={'l1': 1e-5, 'l2': 1e-4}
)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Educational Resources

For more information about neural networks and deep learning:
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
