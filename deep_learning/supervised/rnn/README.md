# Stock Price Prediction using LSTM

This project implements a deep learning model using Long Short-Term Memory (LSTM) networks to predict stock prices. The model is trained on historical Google stock price data and can be used to predict future price movements.

## Features

- Multi-layer LSTM architecture with dropout for better generalization
- Comprehensive data preprocessing and scaling
- Model checkpointing and early stopping
- Detailed logging system
- Visualization of predictions and training metrics
- Modular and maintainable code structure

## Project Structure

```
.
├── data/
│   ├── Google_Stock_Price_Train.csv
│   └── Google_Stock_Price_Test.csv
├── logs/
├── models/
├── plots/
├── src/
│   ├── data_processor.py
│   ├── model.py
│   └── train.py
├── tests/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-price-prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run the training script with the required arguments:

```bash
python src/train.py \
    --train-data data/Google_Stock_Price_Train.csv \
    --test-data data/Google_Stock_Price_Test.csv \
    --epochs 100 \
    --batch-size 32 \
    --sequence-length 60
```

### Arguments

- `--train-data`: Path to the training data CSV file
- `--test-data`: Path to the test data CSV file
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--sequence-length`: Number of time steps to look back (default: 60)

## Model Architecture

The LSTM model consists of:
- 3 LSTM layers with 50 units each
- Dropout layers (0.2) after each LSTM layer
- Final Dense layer for prediction

## Data Format

The input data should be CSV files with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

The model uses the 'Close' price for predictions.

## Results

Training results and visualizations are saved in the `plots/` directory:
- `training_results.png`: Actual vs Predicted prices for training data
- `test_results.png`: Actual vs Predicted prices for test data
- `training_history.png`: Training and validation loss curves

## Logging

Training logs are automatically saved in the `logs/` directory with timestamps.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source: Google Stock Price dataset
- Based on LSTM architecture for time series prediction