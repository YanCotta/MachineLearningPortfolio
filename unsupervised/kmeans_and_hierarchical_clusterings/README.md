<div align="center">

# 🏪 Mall Customer Segmentation Analysis

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*An advanced customer segmentation analysis using unsupervised learning techniques*

[Overview](#-overview) • [Dataset](#-dataset-description) • [Implementation](#%EF%B8%8F-implementation-details) • [Installation](#%EF%B8%8F-installation) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## 📊 Overview

This sophisticated machine learning project harnesses the power of unsupervised clustering techniques to segment mall customers based on their spending patterns and annual income. The analysis delivers actionable insights for retail strategy optimization and enhanced customer relationship management.

## 🔍 Dataset Description

Our analysis leverages `Mall_Customers.csv`, featuring the following key attributes:

| Feature | Description | Type |
|:--------|:------------|:-----|
| 🆔 CustomerID | Unique identifier for each customer | Integer |
| 👤 Genre | Customer's gender | Categorical |
| 📅 Age | Customer's age | Integer |
| 💰 Annual Income (k$) | Yearly income in thousands | Float |
| 🛍️ Spending Score (1-100) | Mall-assigned spending score | Integer |

## 🛠️ Implementation Details

<details>
<summary><b>K-Means Clustering</b></summary>

- ✨ Optimal cluster selection via:
  - 📉 Elbow method with inertia analysis
  - 📊 Silhouette score validation
  - ⚡ K-means++ initialization
- 🔄 Features standardization (StandardScaler)
- 📈 Interactive visualization (matplotlib)

</details>

<details>
<summary><b>Hierarchical Clustering</b></summary>

- 🌳 Agglomerative clustering featuring:
  - 📐 Ward's minimum variance method
  - 📊 Dendrogram visualization
  - 📈 Cophenetic correlation analysis

</details>

## 📈 Key Findings

Our analysis unveiled five distinct customer personas:

| Segment | Profile | Characteristics |
|:--------|:--------|:---------------|
| 🎯 Segment 1 | Premium Customers | High Income, High Spending |
| 💼 Segment 2 | Conservative Spenders | High Income, Low Spending |
| ⚖️ Segment 3 | Average Customers | Average Income, Average Spending |
| 🎊 Segment 4 | Careful Spenders | Low Income, High Spending |
| 🏷️ Segment 5 | Value Seekers | Low Income, Low Spending |

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mall-customer-segmentation.git

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📝 Usage

```bash
# Execute clustering analyses
python src/k_means_clustering.py
python src/hierarchical_clustering.py

# Run tests
pytest tests/
```

## 📦 Dependencies

```plaintext
numpy>=1.19.2
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.3
scipy>=1.6.0
seaborn>=0.11.1
```

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add: AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Yan Cotta](mailto:yanpcotta@gmail.com)**

</div>

