# Binary Image Classification CNN

A professional implementation of a Convolutional Neural Network (CNN) for binary image classification, specifically designed for the cats vs dogs classification task. This project demonstrates industry-standard best practices in deep learning, including modular architecture, comprehensive evaluation metrics, and experiment tracking.

![Sample Predictions](docs/assets/sample_predictions.png)

## Project Overview

This CNN implementation features:
- Modern CNN architecture with batch normalization and dropout
- Comprehensive data augmentation pipeline
- Advanced training features (learning rate scheduling, early stopping)
- Extensive evaluation metrics and visualizations
- TensorBoard integration for experiment tracking
- Modular and maintainable codebase structure

## Dataset

The dataset consists of over 10,000 labeled images of cats and dogs. Due to its size, the dataset is hosted on Google Drive:
[Download Dataset](https://drive.google.com/file/d/1vtO7vdRndB0fCsz6RYCgH6DiYWLY7GUG/view?usp=sharing)

## Project Structure

```
cnn_project/
├── data/                    # Data directory (download dataset here)
├── docs/                    # Documentation and assets
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_processor/     # Data loading and preprocessing
│   ├── model/              # Model architecture
│   ├── evaluation/         # Evaluation metrics and visualization
│   └── training/           # Training pipeline
├── tests/                  # Unit tests
├── main.py                 # Main execution script
├── requirements.txt        # Project dependencies
└── setup.py               # Package installation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cnn_image_classifier.git
cd cnn_image_classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset and extract it to the `data` directory.

## Usage

### Training

Train the model with default parameters:
```bash
python main.py --train-dir data/training_set --test-dir data/test_set
```

Customize training parameters:
```bash
python main.py --train-dir data/training_set \
               --test-dir data/test_set \
               --epochs 50 \
               --batch-size 64 \
               --img-size 128 128 \
               --experiment-name "high_res_training"
```

### Model Evaluation

The training process automatically generates:
- Training history plots
- Confusion matrix
- Sample predictions visualization
- TensorBoard logs

View training progress in TensorBoard:
```bash
tensorboard --logdir logs/fit
```

## Model Architecture

The CNN architecture implements modern best practices:
- Multiple convolutional blocks with increasing filter sizes
- Batch normalization for training stability
- Spatial dropout for regularization
- Dense layers with dropout for classification
- Binary cross-entropy loss with Adam optimizer

## Performance

The model achieves:
- Training accuracy: ~95%
- Validation accuracy: ~92%
- AUC-ROC score: ~0.95

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Yan Cotta

## Acknowledgments

- The dataset used in this project is sourced from [Kaggle's Dogs vs Cats competition](https://www.kaggle.com/c/dogs-vs-cats)
- Architecture inspired by modern CNN implementations in papers [ref1] and [ref2]

## References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)