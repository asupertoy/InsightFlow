import logging
import os
from pathlib import Path
from datetime import datetime

# Define log path
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name="insightflow"):
    """
    Sets up a logger that writes to data/logs/run_{timestamp}.log
    and prints to console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    # File Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name="insightflow"):
    return logging.getLogger(name)
