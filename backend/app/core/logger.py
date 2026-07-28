"""
Structured logging setup. Every agent step, tool call, and error gets logged
with a request/session id so runs are traceable end-to-end (basic observability).
"""
import logging
import sys
import time
import uuid
from functools import wraps

from app.core.config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def log_timing(logger: logging.Logger):
    """Decorator to log execution time of agent tools -- useful for perf debugging."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.info(f"{func.__name__} completed in {elapsed:.1f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"{func.__name__} failed after {elapsed:.1f}ms: {e}")
                raise

        return wrapper

    return decorator
