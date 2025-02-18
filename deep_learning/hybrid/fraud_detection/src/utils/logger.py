import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name):
    """Set up logger with consistent formatting and handling."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.setLevel(logging.INFO)
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler - rotate at 1MB
        os.makedirs('logs', exist_ok=True)
        file_handler = RotatingFileHandler(
            'logs/fraud_detection.log',
            maxBytes=1_000_000,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger