"""Slippage models for realistic execution simulation.

Models the price impact of executing orders in the market, supporting
fixed-percentage, volume-based, and spread-based approaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class SlippageType(str, Enum):
    """Supported slippage calculation methods."""

    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    SPREAD_BASED = "spread_based"


@dataclass
class SlippageModel:
    """Calculate execution slippage for simulated orders.

    Args:
        slippage_type: Method used to estimate slippage.
        fixed_pct: Fixed slippage as a fraction of price (e.g. 0.0005 = 5 bps).
        volume_impact_factor: Multiplier for volume-based slippage.
        spread_pct: Bid-ask spread as a fraction of mid price.
        max_slippage_pct: Hard cap on slippage to prevent unrealistic fills.
    """

    slippage_type: SlippageType = SlippageType.FIXED
    fixed_pct: float = 0.0005
    volume_impact_factor: float = 0.1
    spread_pct: float = 0.0002
    max_slippage_pct: float = 0.01

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        price: float,
        quantity: float,
        *,
        bar_volume: float = 0.0,
        is_buy: bool = True,
    ) -> float:
        """Calculate the slippage-adjusted execution price.

        Args:
            price: Reference price (typically the bar's close).
            quantity: Number of units being traded.
            bar_volume: Volume of the current bar (for volume-based model).
            is_buy: ``True`` for buy orders, ``False`` for sells/shorts.

        Returns:
            The adjusted execution price after slippage.
        """
        slippage_pct = self._raw_slippage(price, quantity, bar_volume)
        slippage_pct = min(slippage_pct, self.max_slippage_pct)

        direction = 1.0 if is_buy else -1.0
        adjusted = price * (1.0 + direction * slippage_pct)

        logger.debug(
            "Slippage: {:.6f}% on price {:.4f} -> {:.4f} ({})",
            slippage_pct * 100.0, price, adjusted,
            "BUY" if is_buy else "SELL",
        )
        return round(adjusted, 6)

    def calculate_cost(
        self,
        price: float,
        quantity: float,
        *,
        bar_volume: float = 0.0,
        is_buy: bool = True,
    ) -> float:
        """Return the dollar cost of slippage for this fill.

        Args:
            price: Reference price.
            quantity: Number of units.
            bar_volume: Volume of the current bar.
            is_buy: ``True`` for buy orders.

        Returns:
            Absolute dollar amount lost to slippage.
        """
        adjusted = self.calculate(
            price, quantity, bar_volume=bar_volume, is_buy=is_buy,
        )
        return abs(adjusted - price) * abs(quantity)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _raw_slippage(
        self,
        price: float,
        quantity: float,
        bar_volume: float,
    ) -> float:
        """Compute raw slippage percentage before capping."""
        if self.slippage_type == SlippageType.FIXED:
            return self.fixed_pct

        if self.slippage_type == SlippageType.VOLUME_BASED:
            if bar_volume <= 0:
                return self.fixed_pct
            participation = abs(quantity) / bar_volume
            return self.fixed_pct + participation * self.volume_impact_factor

        if self.slippage_type == SlippageType.SPREAD_BASED:
            half_spread = self.spread_pct / 2.0
            return half_spread + self.fixed_pct

        return self.fixed_pct

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "slippage_type": self.slippage_type.value,
            "fixed_pct": self.fixed_pct,
            "volume_impact_factor": self.volume_impact_factor,
            "spread_pct": self.spread_pct,
            "max_slippage_pct": self.max_slippage_pct,
        }
