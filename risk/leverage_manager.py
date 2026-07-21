"""Leverage Manager.

Validates the requested leverage multiplier against account bounds.
"""

from __future__ import annotations

from loguru import logger

from config.settings import StrategySettings


class LeverageManager:
    """Validates leverage parameters."""

    def __init__(self, settings: StrategySettings) -> None:
        self.settings = settings

    def validate_leverage(self, requested_leverage: float) -> bool:
        """Ensure requested leverage does not exceed configured maximums.

        Args:
            requested_leverage: Leverage multiplier desired by strategy.

        Returns:
            True if valid, False otherwise.
        """
        if requested_leverage > self.settings.leverage:
            logger.warning(
                "Leverage Reject: Requested ({:.1f}x) exceeds global max ({:.1f}x).",
                requested_leverage,
                self.settings.leverage,
            )
            return False
        
        if requested_leverage < 1.0:
            logger.warning("Leverage Reject: Multiplier cannot be < 1.0")
            return False
            
        return True
