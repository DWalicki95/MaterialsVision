"""
Logging configuration for the materials_vision package.

This module provides centralized logging setup with both console and file
handlers, including log rotation capabilities.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_filename: str = "materials_vision.log"
) -> logging.Logger:
    """
    Configure logging for the materials_vision package.

    Sets up both console and rotating file handlers with consistent formatting.
    The log file uses rotation with a maximum size of 10MB and keeps 5 backup files.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG).
        Default is logging.INFO.
    log_dir : Path, optional
        Directory where log files will be stored. If None, uses the 'logs'
        directory in the project root. Default is None.
    log_filename : str, optional
        Name of the log file. Default is "materials_vision.log".

    Returns
    -------
    logging.Logger
        Configured root logger instance.

    Notes
    -----
    - Log format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    - Date format: '%Y-%m-%d %H:%M:%S'
    - File rotation: 10MB max size, 5 backup files
    - Creates log directory if it doesn't exist

    Examples
    --------
    >>> from materials_vision.logging_config import setup_logging
    >>> import logging
    >>> logger = setup_logging(level=logging.DEBUG)
    >>> logger.info("Logging initialized")

    >>> # Custom log directory
    >>> logger = setup_logging(log_dir=Path("/custom/logs"))
    """
    # Determine log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Get or create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format)
    )

    # File handler with rotation (10MB max size, keep 5 backup files)
    log_file = log_dir / log_filename
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Add both handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return root_logger
