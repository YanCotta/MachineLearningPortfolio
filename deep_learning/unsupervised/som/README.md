# Credit Card Fraud Detection using Self-Organizing Maps (SOM)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional implementation of Self-Organizing Maps for credit card fraud detection, featuring comprehensive data analysis, visualization tools, and a modular codebase structure.

## 🌟 Features

- **Robust Data Processing**
  - Automated data validation and cleaning
  - Advanced feature scaling and normalization
  - Comprehensive error handling

- **Advanced SOM Implementation**
  - Configurable network architecture
  - Automated hyperparameter management
  - Training state tracking
  - Robust anomaly detection algorithms

- **Comprehensive Visualization Suite**
  - Interactive SOM heatmaps
  - Anomaly score distribution analysis
  - Feature importance visualization
  - Automated result plotting and saving

- **Production-Ready Architecture**
  - Modular, object-oriented design
  - Comprehensive error handling
  - Extensive logging
  - Full test coverage
  - Clear documentation

## 📁 Project Structure

```
credit-card-som/
├── dataset/
│   └── Credit_Card_Applications.csv   # Credit card application dataset
├── src/
│   ├── dataset/
│   │   ├── __init__.py               # Dataset module initialization
│   │   └── data_loader.py            # Data loading and preprocessing
│   ├── model/
│   │   ├── __init__.py               # Model module initialization
│   │   └── som_model.py              # SOM implementation
│   └── visualization/
│       ├── __init__.py               # Visualization module initialization
│       └── visualize.py              # Plotting utilities
├── tests/
│   └── test_som.py                   # Unit tests
├── output/                           # Generated visualizations and results
├── notebooks/
│   └── som.ipynb                     # Jupyter notebook implementation
├── main.py                           # Main execution script
├── setup.sh                          # Environment setup script
├── requirements.txt                  # Project dependencies
├── setup.py                          # Package configuration
└── README.md                         # Project documentation
```

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd credit-card-som
   
   # Run setup script
   ./setup.sh
   ```

2. **Run Analysis**:
   ```bash
   python main.py
   ```

   This will:
   - Load and preprocess the credit card data
   - Train the SOM model
   - Generate visualizations in the `output/` directory
   - Save detection results and analysis

## 💻 Usage Examples

### Basic Usage
```python
from src.dataset.data_loader import CreditCardDataLoader
from src.model.som_model import CreditCardSOM

# Load and preprocess data
loader = CreditCardDataLoader('dataset/Credit_Card_Applications.csv')
X_scaled, y, X_original = loader.load_data()

# Initialize and train SOM
som = CreditCardSOM(x=10, y=10, input_len=X_scaled.shape[1])
som.train(X_scaled)

# Detect anomalies
anomalies = som.find_anomalies(X_scaled)
```

### Visualization
```python
from src.visualization.visualize import (
    plot_som_heatmap,
    plot_anomaly_distribution,
    plot_feature_importance
)

# Generate visualizations
winners = som.get_winning_nodes(X_scaled)
distance_map = som.get_feature_map()

# Plot SOM heatmap
plot_som_heatmap(distance_map, winners, y, save_path='output/heatmap.png')
```

## 📊 Model Configuration

The SOM model supports various configuration options:

```python
som = CreditCardSOM(
    x=10,              # Width of the SOM grid
    y=10,              # Height of the SOM grid
    input_len=15,      # Number of input features
    sigma=1.0,         # Neighborhood radius
    learning_rate=0.5  # Learning rate for weight updates
)
```

## 🧪 Testing

Run the test suite:
```bash
python -m unittest discover tests
```

## 📝 Logging

The project uses Python's built-in logging module with the following levels:
- INFO: General execution progress
- WARNING: Non-critical issues
- ERROR: Critical issues that need attention
- DEBUG: Detailed debugging information

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Yan Cotta - yanpcotta@gmail.com

Project Link: [https://github.com/yourusername/credit-card-som](https://github.com/yourusername/credit-card-som)
