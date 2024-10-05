import logging
import time
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)

# Handlers for logging: console and file
stream_handler = logging.StreamHandler()
log_filename = "output.log"
file_handler = logging.FileHandler(filename=log_filename)
handlers = [stream_handler, file_handler]

class TimeFilter(logging.Filter):
    """Custom filter to allow logging only for messages containing 'Running'."""
    
    def filter(self, record):
        return "Running" in record.getMessage()

# Add the custom filter to the logger
logger.addFilter(TimeFilter())

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(asctime)s - %(levelname)s - %(message)s",
    handlers=handlers,
)

def time_logger(func):
    """
    Decorator to log the execution time of a function.

    This decorator logs the start time, end time, and total execution time of the decorated function.
    The function name and execution time are logged at the INFO level.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate execution time
        logger.info(f"Running {func.__name__}: --- {execution_time:.4f} seconds ---")  # Log the execution time
        return result

    return wrapper
