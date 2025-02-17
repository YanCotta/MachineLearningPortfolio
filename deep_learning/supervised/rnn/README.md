# 📈 Stock Price Prediction using LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A sophisticated deep learning solution leveraging LSTM networks for accurate stock price prediction, demonstrated using Google stock data.

## 🌟 Key Features

- **Advanced Architecture**
  - Multi-layer LSTM with dropout
  - Optimized for time series prediction
  - State-of-the-art preprocessing pipeline

- **Production Ready**
  - Model checkpointing
  - Early stopping mechanism
  - Comprehensive logging system
  - Performance visualization suite

## 🏗️ Project Structure

```bash
stock-prediction/
├── 📊 data/
│   ├── Google_Stock_Price_Train.csv
│   └── Google_Stock_Price_Test.csv
├── 📝 logs/
├── 💾 models/
├── 📈 plots/
├── 📁 src/
│   ├── data_processor.py
│   ├── model.py
│   └── train.py
├── 🧪 tests/
├── requirements.txt
└── README.md
```

## 🚀 Quick Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone <url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training Configuration

```bash
python src/train.py \
    --train-data data/Google_Stock_Price_Train.csv \
    --test-data data/Google_Stock_Price_Test.csv \
    --epochs 100 \
    --batch-size 32 \
    --sequence-length 60
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train-data` | Training dataset path | Required |
| `--test-data` | Test dataset path | Required |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--sequence-length` | Lookback window | 60 |

## 🎯 Model Architecture

### Components
- **LSTM Layers**
  - 3x LSTM layers (50 units each)
  - Dropout (0.2) for regularization
  - Dense output layer

## 📊 Data Requirements

### Input Format
```csv
Date,Open,High,Low,Close,Volume
2022-01-01,2500.00,2550.00,2480.00,2520.00,1000000
...
```

## 📈 Performance Visualization

Track model performance through:
- Real-time training metrics
- Prediction accuracy plots
- Loss convergence curves

## 🤝 Contributing

We welcome contributions! Follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. 💫 Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. 📤 Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. 🎁 Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Google Stock Price historical data
- Architecture: Based on state-of-the-art LSTM research
- Community: TensorFlow and Python ecosystems

---
<p align="center">
  <i>Built with 📊 by <a href="https://github.com/YanCotta">Yan Cotta</a></i>
</p>