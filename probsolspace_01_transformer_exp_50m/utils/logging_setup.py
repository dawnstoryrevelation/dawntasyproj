# utils/logging_setup.py
import logging
import sys

def setup_logging(log_level=logging.INFO, rank=0):
    """
    Configures the logger to be clean and informative.
    Only the main process (rank 0) will print logs.
    """
    if rank == 0:
        # If we are the main process, print to stdout
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(handler)
        
        # Set transformers and datasets log levels to be less verbose
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)

    else:
        # For all other processes, disable logging
        logging.disable(logging.CRITICAL)