# Advanced Neural Network Implementation

A production-grade implementation of an Artificial Neural Network (ANN) for binary classification, featuring comprehensive data preprocessing, modular architecture, and advanced evaluation metrics. This implementation focuses on the customer churn prediction problem using the Churn_Modelling dataset.

## 🌟 Features

- **Modular Architecture**
  - Separate modules for data processing, model building, and evaluation
  - Clean, maintainable, and well-documented code
  - Type hints and comprehensive error handling
  - Flexible model architecture configuration

- **Advanced Data Preprocessing**
  - Automated handling of categorical variables with label encoding
  - Feature scaling using StandardScaler
  - Flexible train/validation/test splitting options
  - Comprehensive data validation and error handling
  - Support for missing value handling

- **Sophisticated Model Architecture**
  - Fully configurable layer architecture with flexible depth and width
  - Advanced regularization techniques:
    - Dropout layers with configurable rates
    - BatchNormalization for training stability
    - L1/L2 regularization with customizable parameters
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping to prevent overfitting
  - Model checkpointing for best weights

- **Comprehensive Evaluation**
  - Multiple performance metrics:
    - Accuracy, AUC-ROC, Precision, Recall
    - Detailed classification report
    - Confusion matrix visualization
  - Advanced visualization tools:
    - Training history plots
    - ROC and Precision-Recall curves
    - Learning rate analysis
  - Detailed prediction analysis with feature importance

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.8+
- scikit-learn 1.0+
- pandas
- numpy
- matplotlib
- seaborn
Additional dependencies are listed in `requirements.txt`

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd ann
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
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
│   └── Churn_Modelling.csv    # Customer churn dataset
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Data preprocessing and validation
│   ├── model_builder.py       # Neural network architecture
│   ├── evaluation.py          # Model evaluation and visualization
│   └── main.py               # Main execution script
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model_builder.py
│   └── test_evaluation.py
├── requirements.txt
└── README.md
```

## 📊 Model Architecture

The neural network implements a binary classification model with:
- Configurable hidden layers with ReLU activation
- Optional Dropout and BatchNormalization for each layer
- Customizable L1/L2 regularization
- Binary cross-entropy loss function
- Adam optimizer with configurable learning rate
- Multiple evaluation metrics tracking

Default architecture (customizable in main.py):
```python
model = model_builder.build_model(
    hidden_layers=[32, 16, 8],
    dropout_rates=[0.3, 0.2, 0.1],
    use_batch_norm=True,
    kernel_regularizer={'l1': 1e-5, 'l2': 1e-4}
)
```

## 📈 Performance Metrics

The model evaluation includes:
- Classification accuracy
- ROC-AUC score
- Precision and recall metrics
- F1 score
- Detailed confusion matrix
- Feature importance analysis
- Training/validation learning curves
- ROC and Precision-Recall curves

## 🔧 Training Configuration

You can customize the training process in `main.py`:
```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=10),
        ReduceLROnPlateau(factor=0.1, patience=5),
        ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
    ]
)
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
pytest tests/
```

The test suite includes:
- Data preprocessing validation
- Model architecture testing
- Training process verification
- Evaluation metrics validation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Educational Resources

For more information about neural networks and deep learning:
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
