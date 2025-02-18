from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stock_prediction_rnn",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.8.0',
        'numpy>=1.19.2',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'typing-extensions>=3.7.4',
        'pytest>=6.0.0',
        'python-dateutil>=2.8.1'
    ],
    python_requires='>=3.8',
    author="Yan Cotta",
    author_email="yanpcotta@gmail.com",
    description="Advanced RNN-based stock price prediction with comprehensive evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="deep learning, RNN, LSTM, stock prediction, time series",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        'console_scripts': [
            'train-stock-predictor=src.train:main',
        ],
    }
)