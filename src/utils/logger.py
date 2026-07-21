"""
Logging Configuration – StartupPulse AI.

Provides a centralized logging setup with console and optional file handler.
All modules import and use ``get_logger(__name__)`` for consistent output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import ROOT

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """Configure root logger with console and optional file handler.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path to a log file. If None, only console output.
    """
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is None:
        log_file = ROOT / "logs" / "startuppulse.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FMT,
        handlers=handlers,
        force=True,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured first.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        A configured Logger instance.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
