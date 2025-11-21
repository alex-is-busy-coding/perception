import logging
from rich.logging import RichHandler

def setup_logging(log_level: str = "INFO"):
    """
    Configures the root logger with RichHandler.
    
    Args:
        log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING").
    """
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )