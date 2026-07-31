"""Structured logging with secret redaction."""

import logging
import re
from collections.abc import Mapping
from typing import Any

_SECRET_KEY_PATTERN: re.Pattern[str] = re.compile(r"(?i)(key|token|secret|password|credential)")
_REDACTED: str = "[REDACTED]"

DEFAULT_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def is_secret_key(key: str) -> bool:
    """Return True if the key name matches a secret pattern."""
    return bool(_SECRET_KEY_PATTERN.search(key))


def redact(data: Any) -> Any:
    """Recursively redact values whose keys match secret patterns.

    Mappings have their secret-keyed values replaced with ``[REDACTED]``.
    Lists are traversed element-wise. All other values pass through unchanged.
    """
    if isinstance(data, Mapping):
        return {k: (_REDACTED if is_secret_key(str(k)) else redact(v)) for k, v in data.items()}
    if isinstance(data, list):
        return [redact(item) for item in data]
    return data


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the rag-evals logger."""
    logger = logging.getLogger("rag_evals")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str = "rag_evals") -> logging.Logger:
    """Return a child logger under the rag-evals namespace."""
    return logging.getLogger(name)
