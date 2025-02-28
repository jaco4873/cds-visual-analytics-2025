"""
Utility functions for logging.
"""

import logging


def get_logger() -> logging.Logger:
    """
    Root logger for the shared library.
    """
    logger = logging.getLogger()

    if not logger.handlers:
        # Set logging level
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger


# Create a top level logger
logger = get_logger()
