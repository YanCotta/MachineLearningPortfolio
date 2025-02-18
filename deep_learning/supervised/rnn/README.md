# 🚀 Advanced Stock Price Prediction with RNN

A sophisticated implementation of a Recurrent Neural Network (RNN) for stock price prediction, featuring bidirectional LSTM layers, comprehensive technical indicators, and advanced evaluation metrics.

## 🌟 Features

- **Advanced Model Architecture**
  - Bidirectional LSTM layers for better temporal pattern recognition
  - Configurable multi-layer architecture
  - Batch normalization for training stability
  - Dropout regularization for preventing overfitting
  - Adaptive learning rate scheduling

- **Technical Analysis Integration**
  - Moving averages (MA7, MA20)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Customizable feature selection

- **Comprehensive Evaluation**
  - Multiple performance metrics (MSE, RMSE, MAE, R², MAPE)
  - Direction accuracy analysis
  - Error distribution visualization
  - Training history plots
  - Detailed evaluation reports

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.8+
- Additional dependencies in `requirements.txt`

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd rnn
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python src/train.py \
    --train-file dataset/Google_Stock_Price_Train.csv \
    --test-file dataset/Google_Stock_Price_Test.csv \
    --sequence-length 60 \
    --epochs 100 \
    --batch-size 32 \
    --use-features
```

## 🏗️ Project Structure

```
rnn/
├── dataset/                    # Stock price datasets
│   ├── Google_Stock_Price_Train.csv
│   └── Google_Stock_Price_Test.csv
├── src/                       # Source code
│   ├── model.py              # LSTM model architecture
│   ├── data_processor.py     # Data preprocessing pipeline
│   ├── evaluation.py         # Evaluation metrics & visualization
│   └── train.py             # Training script
├── tests/                    # Unit tests
│   ├── test_model.py
│   ├── test_data_processor.py
│   └── test_evaluation.py
├── requirements.txt          # Project dependencies
├── setup.py                 # Package configuration
└── README.md               # Documentation
```

## 🔧 Advanced Configuration

### Model Architecture
```python
model = StockPredictor(
    sequence_length=60,
    n_features=6,  # When using technical indicators
    lstm_units=[100, 50, 50],
    dropout_rates=[0.3, 0.2, 0.2],
    bidirectional=True,
    use_batch_norm=True,
    learning_rate=0.001
)
```

### Feature Selection
```python
features = [
    'Close',      # Stock closing price
    'MA7',        # 7-day moving average
    'MA20',       # 20-day moving average
    'RSI',        # Relative Strength Index
    'MACD',       # Moving Average Convergence Divergence
    'BB_middle'   # Bollinger Band middle line
]
```

## 📊 Evaluation Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Direction Accuracy

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🔍 Model Output

The model generates:
- Trained model weights
- Training history plots
- Prediction vs actual plots
- Error distribution analysis
- Comprehensive evaluation report

## 📈 Sample Results

Check the `experiments/` directory after training for:
- Training history visualization
- Prediction plots
- Error analysis
- Performance metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

1. [LSTM Networks for Time Series Prediction](https://arxiv.org/abs/1902.10877)
2. [Technical Analysis in Financial Markets](https://www.sciencedirect.com/science/article/abs/pii/S0927539804000829)
3. [Deep Learning for Time Series Forecasting](https://arxiv.org/abs/2004.13408)