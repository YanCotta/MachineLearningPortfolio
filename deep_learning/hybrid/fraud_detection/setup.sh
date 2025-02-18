#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data output logs

# Download dataset if not present
if [ ! -f "data/Credit_Card_Applications.csv" ]; then
    echo "Please place the Credit_Card_Applications.csv file in the data directory"
fi

echo "Setup complete! You can now run 'python main.py' to start the analysis."