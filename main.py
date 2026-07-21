"""Market Regime Bot — Phase 1 entry point.

This script bootstraps the application:

1. Loads and validates configuration.
2. Initialises the centralised logger.
3. Downloads, cleans, and validates OHLCV data.
4. Prints the validation report to the console.

Usage::

    python main.py                        # default config
    python main.py --config path/to.yaml  # custom config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with an optional ``config`` path.
    """
    parser = argparse.ArgumentParser(
        description="Market Regime Bot — Phase 1: Data Collection",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.yaml file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Phase 1 data-collection pipeline."""
    args = parse_args()

    # ---- Configuration ----------------------------------------------------
    from config.settings import get_settings

    try:
        settings = get_settings(config_path=args.config)
    except Exception as exc:
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Logging ----------------------------------------------------------
    from utils.logger import setup_logger

    setup_logger(settings.log_dir)

    logger.info("=" * 55)
    logger.info("  Market Regime Bot — Phase 1: Data Collection")
    logger.info("=" * 55)
    logger.info("Broker      : {}", settings.broker.value)
    logger.info("Symbol      : {}", settings.symbol)
    logger.info("Timeframe   : {}", settings.timeframe.value)
    logger.info("Date range  : {} → {}", settings.start_date, settings.end_date)
    logger.info("Capital     : ${:,.2f}", settings.initial_capital)
    logger.info("Cache       : {}", "enabled" if settings.use_cache else "disabled")
    logger.info("=" * 55)

    # ---- Data pipeline ----------------------------------------------------
    from data_loader.data_manager import DataManager

    manager = DataManager(settings)

    try:
        df, report = manager.get_data()
    except Exception as exc:
        logger.exception("Data pipeline failed: {}", exc)
        sys.exit(1)

    # ---- Report -----------------------------------------------------------
    logger.info("\n{}", report.summary())

    if report.passed:
        logger.success(
            "Phase 1 complete — {} clean rows saved to disk.",
            len(df),
        )
    else:
        logger.warning(
            "Phase 1 complete with validation warnings — review the "
            "report above before proceeding to Phase 2.",
        )

    # Quick data preview.
    logger.info("Data preview (first 5 rows):\n{}", df.head().to_string())
    logger.info("Data preview (last 5 rows):\n{}", df.tail().to_string())


if __name__ == "__main__":
    main()
