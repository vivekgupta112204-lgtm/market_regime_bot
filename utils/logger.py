"""Centralised logging configuration built on Loguru.

Call ``setup_logger()`` once at application start.  Every module can then
simply do::

    from loguru import logger

and all sinks (console + rotating file) will already be attached.

Design decisions
~~~~~~~~~~~~~~~~
* **Rotation**: Log files rotate at 50 MB to avoid unbounded growth.
* **Retention**: Old files are kept for 30 days.
* **Compression**: Rotated files are gzip-compressed to save disk space.
* **Coloured console**: Rich colour output is enabled for developer
  ergonomics; it degrades gracefully in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_dir: Path,
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "50 MB",
    retention: str = "30 days",
    compression: str = "gz",
) -> None:
    """Configure Loguru sinks for the entire application.

    Args:
        log_dir: Directory where log files will be written.
        console_level: Minimum level for the stderr sink.
        file_level: Minimum level for the file sink.
        rotation: Size or time string that triggers file rotation.
        retention: How long rotated files are kept.
        compression: Compression algorithm for rotated files.
    """
    # Remove the default Loguru handler so we control everything.
    logger.remove()

    # ---- Console sink -----------------------------------------------------
    logger.add(
        sys.stderr,
        level=console_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ---- Rotating file sink -----------------------------------------------
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "market_regime_bot.log"

    logger.add(
        str(log_file),
        level=file_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} — {message}"
        ),
        rotation=rotation,
        retention=retention,
        compression=compression,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        "Logger initialised — console={}, file={}, log_dir={}",
        console_level,
        file_level,
        log_dir,
    )
