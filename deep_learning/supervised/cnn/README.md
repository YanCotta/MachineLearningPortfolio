# 🔍 Binary Image Classification CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A state-of-the-art CNN implementation for binary image classification, showcasing industry-standard deep learning practices through a cats vs. dogs classification task.

## 🌟 Highlights

- **Advanced Architecture**
  - Modern CNN with ResNet-style skip connections
  - Batch normalization and advanced regularization
  - Mixed precision training support
  - Multi-GPU compatibility
  - Configurable model parameters

- **Production Features**
  - Advanced data augmentation with albumentations
  - Mixed precision training
  - Multi-threaded data loading
  - Memory-efficient data generators
  - Automatic dataset validation
  - Comprehensive error handling

- **Experiment Tracking**
  - Weights & Biases integration
  - MLflow support
  - TensorBoard monitoring
  - Advanced metric logging
  - Training history visualization

## 📊 Model Features

- **Architecture**
  - Configurable convolutional blocks
  - Skip connections for better gradient flow
  - Advanced regularization (L2, Dropout, BatchNorm)
  - Mixed precision support

- **Training**
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Experiment tracking
  - Training resumption support

- **Evaluation**
  - Comprehensive metrics (Accuracy, AUC, F1)
  - Confusion matrix visualization
  - Prediction visualization
  - Training history plots

## 🏗️ Project Structure

```bash
src/
├── data_processor/          # Data loading and preprocessing
│   ├── data_loader.py      # Advanced data loading with augmentation
│   └── __init__.py
├── model/                  # Model architecture
│   ├── cnn_model.py       # Modern CNN implementation
│   └── __init__.py
├── training/              # Training pipeline
│   ├── trainer.py        # Training with experiment tracking
│   └── __init__.py
└── evaluation/           # Evaluation tools
    ├── evaluator.py     # Metrics and visualization
    └── __init__.py
```

## 🔧 Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone <repository-url>

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

### Configuration Options
- `--train-dir`: Training data directory
- `--test-dir`: Test data directory
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--img-size`: Input image dimensions
- `--experiment-name`: Name for tracking experiments

## 📈 Monitoring

Monitor training in real-time through multiple platforms:
- **TensorBoard**: `tensorboard --logdir logs/fit`
- **Weights & Biases**: View experiments at wandb.ai
- **MLflow**: Track experiments locally or on your MLflow server

## 🤝 Contributing

See our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style
- Development process
- Test requirements
- Pull request process

## 📚 References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)
3. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
4. [Albumentations: Fast and Flexible Image Augmentations](https://arxiv.org/abs/1809.06839)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Kaggle's Dogs vs Cats competition](https://www.kaggle.com/c/dogs-vs-cats)
- Architecture inspiration: ResNet and VGGNet

---
<p align="center">
  <i>Developed with 💻 by <a href="https://github.com/YanCotta">Yan Cotta</a></i>
</p>