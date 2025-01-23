# Mall Customer Segmentation Analysis 🛍️

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

This advanced machine learning project implements unsupervised clustering techniques to segment mall customers based on their spending patterns and annual income. The analysis provides valuable insights for retail strategy and customer relationship management.

## 🔍 Dataset Description

The analysis utilizes `Mall_Customers.csv` with the following features:

| Feature | Description | Type |
|---------|------------|------|
| CustomerID | Unique identifier for each customer | Integer |
| Genre | Customer's gender | Categorical |
| Age | Customer's age | Integer |
| Annual Income (k$) | Yearly income in thousands | Float |
| Spending Score (1-100) | Mall-assigned spending score | Integer |

## 🛠️ Implementation Details

### K-Means Clustering
- Implements optimal cluster selection using:
  - Elbow method with inertia analysis
  - Silhouette score validation
  - K-means++ initialization for enhanced convergence
- Features standardization using StandardScaler
- Interactive visualization using matplotlib

### Hierarchical Clustering
- Implements agglomerative clustering with:
  - Ward's minimum variance method
  - Dendrogram analysis for optimal cluster determination
  - Cophenetic correlation coefficient calculation

## 📈 Key Findings

The analysis reveals five distinct customer segments:
1. High Income, High Spending
2. High Income, Low Spending
3. Average Income, Average Spending
4. Low Income, High Spending
5. Low Income, Low Spending

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mall-customer-segmentation.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📝 Usage

```bash
# Run K-means clustering analysis
python k_means_clustering.py

# Run Hierarchical clustering analysis
python hierarchical_clustering.py
```

## Testing
To run all tests:
```bash
python -m unittest discover tests
```

# K-Means & Hierarchical Clustering

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
To run clustering scripts:
```bash
python src/k_means_clustering.py
python src/hierarchical_clustering.py
```

## Tests
```bash
pytest tests/
```

## License
This sub-project is covered under the [MIT License](../../../../LICENSE).

## 📦 Requirements

```
numpy>=1.19.2
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.3
scipy>=1.6.0
seaborn>=0.11.1
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

yanpcotta@gmail.com

