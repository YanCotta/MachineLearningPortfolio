from setuptools import setup, find_packages

setup(
    name="ann",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
    author="Yan",
    description="Artificial Neural Network Implementation",
    python_requires=">=3.7",
)
