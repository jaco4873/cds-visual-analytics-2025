"""
Logger utility for application-wide logging.
"""

import logging
import sys


def get_logger(
    name: str = "application_logger",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Name of the logger
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Default application logger instance
logger = get_logger()
