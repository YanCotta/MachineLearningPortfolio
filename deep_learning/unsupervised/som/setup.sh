#!/bin/bash

# Setup script for Credit Card Fraud Detection SOM project

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Setting up project structure..."
mkdir -p output
mkdir -p tests/__pycache__
mkdir -p src/__pycache__

# Run tests
echo "Running tests..."
python -m unittest discover tests

# Print setup completion message
echo "Setup complete! You can now run the project using:"
echo "python main.py"