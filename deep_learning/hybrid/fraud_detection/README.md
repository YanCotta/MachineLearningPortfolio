# 🔍 Hybrid Fraud Detection System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated fraud detection system that combines unsupervised and supervised deep learning approaches. The system first uses Self-Organizing Maps (SOM) to detect anomalous patterns in credit card applications, then trains an Artificial Neural Network (ANN) to learn and classify these patterns.

## 🌟 Features

- **Multi-Modal Learning Approach**
  - Unsupervised pattern detection using SOM
  - Supervised classification using ANN
  - Transfer of knowledge between models
  - Automatic threshold adjustment
  
- **Advanced Data Processing**
  - Automated feature scaling optimized for each model
  - Comprehensive data validation and cleaning
  - Intelligent missing value handling
  - Automatic infinite value detection

- **Sophisticated Visualization Suite**
  - Interactive SOM heatmaps with fraud overlays
  - Feature importance analysis with sorted rankings
  - Detailed fraud pattern distribution analysis
  - Comprehensive performance metrics visualization
  - High-resolution outputs (300 DPI)

- **Production-Ready Architecture**
  - Modular, object-oriented design
  - Comprehensive error handling and validation
  - Advanced logging with rotation
  - GPU acceleration support
  - Memory-efficient processing

## 🏗️ Project Structure

```
fraud_detection/
├── data/
│   └── Credit_Card_Applications.csv
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── data_loader.py        # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── som_detector.py       # SOM implementation
│   │   └── ann_classifier.py     # ANN implementation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualizer.py         # Plotting utilities
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Logging configuration
├── output/                       # Generated visualizations
├── logs/                        # Application logs
├── main.py                      # Main execution script
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   python main.py
   ```

   This will:
   - Load and preprocess the credit card data
   - Train the SOM for initial fraud detection
   - Use SOM results to train the ANN
   - Generate visualizations in the `output/` directory
   - Save detection results and analysis

## 📊 How It Works

1. **Unsupervised Learning Phase (SOM)**
   - Identifies suspicious patterns in data
   - Creates topology-preserving map of credit card applications
   - Detects anomalous cases based on neighborhood distances
   - Automatically adjusts detection threshold if needed

2. **Supervised Learning Phase (ANN)**
   - Takes SOM-detected patterns as training input
   - Dynamic architecture based on input dimensions
   - Uses dropout and batch normalization for robust learning
   - Early stopping to prevent overfitting

3. **Visualization & Analysis**
   - Generates high-resolution SOM heatmaps showing fraud clusters
   - Plots feature distributions for detected frauds
   - Shows sorted feature importance in classification
   - Provides detailed model performance metrics

## 📈 Performance Features

- **SOM Detection**
  - Adaptive threshold adjustment
  - Quantization error-based anomaly scoring
  - Topology-preserving feature mapping
  - Automatic hyperparameter optimization

- **ANN Classification**
  - Dynamic architecture scaling
  - Gradient-based feature importance
  - Advanced regularization techniques
  - GPU acceleration support

- **Visualization**
  - High-resolution outputs (300 DPI)
  - Comprehensive metric logging
  - Automated report generation
  - Interactive plot capabilities

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.