# 🔍 Hybrid Fraud Detection System

A sophisticated fraud detection system that combines unsupervised and supervised deep learning approaches. The system first uses Self-Organizing Maps (SOM) to detect anomalous patterns in credit card applications, then trains an Artificial Neural Network (ANN) to learn and classify these patterns.

## 🌟 Features

- **Multi-Modal Learning Approach**
  - Unsupervised pattern detection using SOM
  - Supervised classification using ANN
  - Transfer of knowledge between models
  
- **Advanced Data Processing**
  - Automated feature scaling
  - Separate preprocessing for SOM and ANN
  - Comprehensive data validation

- **Sophisticated Visualization Suite**
  - SOM heatmaps with fraud overlays
  - Feature importance analysis
  - Fraud pattern distribution plots
  - Model performance metrics

- **Production-Ready Architecture**
  - Modular, object-oriented design
  - Comprehensive error handling
  - Advanced logging system
  - Clear documentation

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

2. **Supervised Learning Phase (ANN)**
   - Takes SOM-detected patterns as training input
   - Learns to classify between normal and fraudulent patterns
   - Uses dropout and batch normalization for robust learning

3. **Visualization & Analysis**
   - Generates SOM heatmaps showing fraud clusters
   - Plots feature distributions for detected frauds
   - Shows feature importance in classification
   - Provides detailed model performance metrics

## 📈 Performance Metrics

- SOM detection captures subtle anomalies in high-dimensional data
- ANN achieves high precision in fraud classification
- Combined approach reduces false positives
- Visualization tools enable intuitive pattern analysis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.