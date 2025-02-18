from setuptools import setup, find_packages

setup(
    name="credit-card-som",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "minisom>=2.2.0"
    ],
    author="Yan Cotta",
    author_email="yanpcotta@gmail.com",
    description="A Self-Organizing Map implementation for credit card fraud detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)