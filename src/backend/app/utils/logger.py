# src/backend/app/utils/logger.py
import logging
import sys
from typing import Optional
from ..config import settings

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set level
        log_level = level or settings.LOG_LEVEL
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger