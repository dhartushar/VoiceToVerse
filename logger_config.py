import logging

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers in case of multiple imports
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler
    logger.addHandler(file_handler)