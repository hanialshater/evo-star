# -*- coding: utf-8 -*-
import logging
import sys


def setup_logger(level=logging.INFO):
    """
    Configures and returns a root logger for the application.
    """
    logger = logging.getLogger("EvoAgent")

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Quiet down the noisy Google API libraries
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)

    logger.info(
        f"Logger '{logger.name}' configured with level {logging.getLevelName(level)}."
    )
    return logger


# Default logger instance to be imported by other modules
logger = logging.getLogger("EvoAgent")

print("alpha_evolve_framework/logging_utils.py created.")
