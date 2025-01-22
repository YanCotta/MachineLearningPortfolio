# Market Basket Analysis Framework 📊

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/stable/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

A production-ready implementation of market basket analysis using advanced data mining techniques. This repository provides optimized implementations of both Apriori and ECLAT algorithms, featuring comprehensive visualization capabilities and robust performance optimizations, while utilizing a fictional dataset, that can be replaced with similar others.

## 🎯 Core Features

### Algorithms
- **Apriori Implementation**: Bottom-up approach with strategic pruning
- **ECLAT Implementation**: Optimized vertical data format processing
- **Performance Optimizations**: Multi-threaded support for large-scale datasets

### Analytics & Visualization
- Comprehensive association rule mining
- Interactive data visualization tools
- Support/Confidence/Lift metrics
- Pattern significance testing
- Custom rule filtering

## ⚙️ Technical Requirements

### System Prerequisites
- Python 3.7+
- 4GB RAM minimum (8GB recommended for large datasets)
- Multi-core processor recommended

### Dependencies
```bash
pandas>=1.2.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
apyori>=1.1.2
```

## Performance

| Dataset Size | Transactions | Processing Time |
|-------------|--------------|-----------------|
| Small       | 1,000        | 0.8s           |
| Medium      | 10,000       | 2.5s           |
| Large       | 100,000      | 12.3s          |

## Usage
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Apriori analysis:
   ```bash
   python src/market_with_apriori.py
   ```
3. Run ECLAT analysis:
   ```bash
   python src/market_with_eclat.py
   ```
4. Test the project:
   ```bash
   python -m unittest discover -s tests
   ```

## Contributing
Feel free to contribute as much as you want!!

## Author
**Yan Cotta**
- Email: yanpcotta@gmail.com
- [LinkedIn](https://linkedin.com/in/yan-cotta)
- [GitHub](https://github.com/YanCotta)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Apyori library team
- scikit-learn community