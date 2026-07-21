"""Commission and fee models for realistic trade-cost simulation.

Supports flat-fee, percentage-based, and tiered commission schedules,
as well as exchange fees and borrow costs for short positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class CommissionType(str, Enum):
    """Supported commission calculation methods."""

    FLAT = "flat"
    PERCENTAGE = "percentage"
    TIERED = "tiered"


@dataclass(frozen=True)
class CommissionTier:
    """A single tier in a tiered commission schedule.

    Attributes:
        max_notional: Upper bound of the tier (inclusive).  Use ``float('inf')``
            for the final catch-all tier.
        rate: Commission rate applied within this tier (as a decimal, e.g.
            0.001 for 0.1 %).
    """

    max_notional: float
    rate: float


@dataclass
class CommissionModel:
    """Calculate trading commissions and ancillary fees.

    Args:
        commission_type: The primary commission calculation method.
        flat_fee: Per-trade flat fee (used when *commission_type* is ``FLAT``).
        percentage_rate: Percentage of notional value (used when *commission_type*
            is ``PERCENTAGE``).
        tiers: Ordered list of :class:`CommissionTier` objects (used when
            *commission_type* is ``TIERED``).
        exchange_fee_rate: Exchange/clearing fee as a fraction of notional.
        borrow_fee_annual: Annualised borrow rate for short positions.
        min_commission: Minimum commission per trade.
    """

    commission_type: CommissionType = CommissionType.PERCENTAGE
    flat_fee: float = 1.0
    percentage_rate: float = 0.001
    tiers: list[CommissionTier] = field(default_factory=list)
    exchange_fee_rate: float = 0.0
    borrow_fee_annual: float = 0.0
    min_commission: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        quantity: float,
        price: float,
        *,
        is_short: bool = False,
        holding_days: int = 0,
    ) -> float:
        """Calculate total commission + fees for a single fill.

        Args:
            quantity: Number of units traded.
            price: Execution price per unit.
            is_short: Whether this is a short-sale trade.
            holding_days: Days the short position was held (for borrow fees).

        Returns:
            Total cost in fiat currency.
        """
        notional = abs(quantity * price)
        commission = self._base_commission(notional)
        commission = max(commission, self.min_commission)

        exchange_fee = notional * self.exchange_fee_rate
        borrow_fee = self._borrow_fee(notional, holding_days) if is_short else 0.0

        total = commission + exchange_fee + borrow_fee
        logger.debug(
            "Commission: {:.4f} (base={:.4f}, exch={:.4f}, borrow={:.4f}) "
            "on notional {:.2f}",
            total, commission, exchange_fee, borrow_fee, notional,
        )
        return round(total, 6)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _base_commission(self, notional: float) -> float:
        """Compute base commission from the chosen method."""
        if self.commission_type == CommissionType.FLAT:
            return self.flat_fee

        if self.commission_type == CommissionType.PERCENTAGE:
            return notional * self.percentage_rate

        if self.commission_type == CommissionType.TIERED:
            return self._tiered_commission(notional)

        return 0.0

    def _tiered_commission(self, notional: float) -> float:
        """Walk through tiers and accumulate commission."""
        if not self.tiers:
            return notional * self.percentage_rate

        remaining = notional
        total = 0.0
        prev_upper = 0.0

        for tier in sorted(self.tiers, key=lambda t: t.max_notional):
            tier_width = tier.max_notional - prev_upper
            applicable = min(remaining, tier_width)
            total += applicable * tier.rate
            remaining -= applicable
            prev_upper = tier.max_notional
            if remaining <= 0:
                break

        if remaining > 0:
            total += remaining * self.tiers[-1].rate

        return total

    def _borrow_fee(self, notional: float, holding_days: int) -> float:
        """Calculate borrow cost for short positions."""
        if holding_days <= 0 or self.borrow_fee_annual <= 0:
            return 0.0
        daily_rate = self.borrow_fee_annual / 365.0
        return notional * daily_rate * holding_days

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "commission_type": self.commission_type.value,
            "flat_fee": self.flat_fee,
            "percentage_rate": self.percentage_rate,
            "exchange_fee_rate": self.exchange_fee_rate,
            "borrow_fee_annual": self.borrow_fee_annual,
            "min_commission": self.min_commission,
        }
