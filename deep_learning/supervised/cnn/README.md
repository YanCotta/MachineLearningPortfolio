# 🔍 Binary Image Classification CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A state-of-the-art CNN implementation for binary image classification, showcasing industry-standard deep learning practices through a cats vs. dogs classification task.

## 🌟 Highlights

- **Modern Architecture**
  - Advanced CNN design with batch normalization
  - Dropout regularization for better generalization
  - State-of-the-art optimization techniques

- **Production Quality**
  - Comprehensive data augmentation pipeline
  - Learning rate scheduling & early stopping
  - TensorBoard integration for monitoring
  - Modular, maintainable codebase

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~92% |
| AUC-ROC Score | ~0.95 |

## 🏗️ Project Structure

```bash
cnn_project/
├── 📁 data/                # Dataset storage
├── 📁 docs/                # Documentation & assets
├── 📁 notebooks/           # Jupyter notebooks
├── 📁 src/                 # Source code
│   ├── data_processor/     # Data pipeline
│   ├── model/             # CNN architecture
│   ├── evaluation/        # Metrics & visualization
│   └── training/          # Training pipeline
├── 📁 tests/              # Unit tests
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── setup.py              # Package config
```

## 🔧 Quick Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Virtual environment (recommended)

### Installation
```bash
# Clone repository
git clone 

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset
📥 [Download Dataset](https://drive.google.com/file/d/1vtO7vdRndB0fCsz6RYCgH6DiYWLY7GUG/view?usp=sharing)
> 10,000+ labeled images of cats and dogs for training and validation

## 🚀 Usage

### Basic Training
```bash
python main.py --train-dir data/training_set --test-dir data/test_set
```

### Advanced Configuration
```bash
python main.py --train-dir data/training_set \
               --test-dir data/test_set \
               --epochs 50 \
               --batch-size 64 \
               --img-size 128 128 \
               --experiment-name "high_res_training"
```

## 🎛️ Model Architecture

### Key Components
- **Convolutional Blocks**
  - Multiple layers with increasing filters
  - Batch normalization for stability
  - Spatial dropout for regularization

- **Classification Head**
  - Dense layers with dropout
  - Binary cross-entropy loss
  - Adam optimizer

## 📈 Training Progress

Monitor training in real-time:
```bash
tensorboard --logdir logs/fit
```

## 🤝 Contributing

We welcome contributions! Here's how:

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

## 📚 References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Kaggle's Dogs vs Cats competition](https://www.kaggle.com/c/dogs-vs-cats)
- Architecture inspiration: ResNet and VGGNet

---
<p align="center">
  <i>Developed with 💻 by <a href="https://github.com/YanCotta">Yan Cotta</a></i>
</p>