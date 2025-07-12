import time
import logging
import torch
from datetime import timedelta

def setup_logger(name: str) -> logging.Logger:
    """
    Create a console logger with INFO level and a standard formatter.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

def timeit(method):
    """
    Decorator to measure execution time of a function.
    """
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print(f"{method.__name__} 耗时: {timedelta(seconds=end-start)}")
        return result
    return timed

def log_gpu_memory(logger: logging.Logger):
    """
    Log the current GPU memory allocation (in GB) if CUDA is available.
    """
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024 ** 3
        logger.info(f"GPU 显存占用: {mem:.2f} GB")
