# Credit Card Fraud Detection using Self-Organizing Maps (SOM)

This project implements a Self-Organizing Map (SOM) neural network for detecting potential fraudulent credit card applications. The SOM is an unsupervised deep learning algorithm that helps identify patterns and anomalies in high-dimensional data.

## Features

- Modular implementation of SOM for credit card fraud detection
- Advanced data preprocessing and scaling
- Comprehensive visualization tools:
  - SOM heatmaps with mapped data points
  - Anomaly score distributions
  - Feature importance analysis
- Configurable model parameters
- Production-ready code structure

## Project Structure

```
credit-card-som/
├── dataset/
│   └── Credit_Card_Applications.csv
├── src/
│   ├── dataset/
│   │   └── data_loader.py
│   ├── model/
│   │   └── som_model.py
│   └── visualization/
│       └── visualize.py
├── notebooks/
│   └── som.ipynb
├── main.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone 
cd {path}
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

```bash
python main.py
```

This will:
1. Load and preprocess the credit card application data
2. Train the SOM model
3. Generate visualizations
4. Output detected potential fraudulent applications

### Using as a Package

```python
from src.dataset.data_loader import CreditCardDataLoader
from src.model.som_model import CreditCardSOM
from src.visualization.visualize import plot_som_heatmap

# Load and preprocess data
loader = CreditCardDataLoader('path/to/data.csv')
X_scaled, y, X_original = loader.load_data()

# Initialize and train SOM
som = CreditCardSOM(x=10, y=10, input_len=15)
som.train(X_scaled)

# Detect anomalies
anomalies = som.find_anomalies(X_scaled)
```

## Model Configuration

The SOM model can be configured with the following parameters:

- `x`, `y`: Dimensions of the SOM grid (default: 10x10)
- `input_len`: Number of input features (default: 15)
- `sigma`: Radius of the neighborhood function (default: 1.0)
- `learning_rate`: Learning rate for weight updates (default: 0.5)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
