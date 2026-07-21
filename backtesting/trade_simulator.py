"""Trade simulation engine for backtesting.

Simulates order execution with realistic market, limit, and stop orders,
including partial fills, order rejections, and cost application via
the commission and slippage models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from backtesting.commission_model import CommissionModel
from backtesting.slippage_model import SlippageModel


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "BUY"
    SELL = "SELL"


class SimOrderType(str, Enum):
    """Supported order types for simulation."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class FillStatus(str, Enum):
    """Outcome of a simulated order."""

    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


@dataclass
class SimulatedOrder:
    """A backtest order awaiting execution.

    Attributes:
        symbol: Instrument ticker.
        side: Buy or sell.
        order_type: Market, limit, or stop.
        quantity: Desired number of units.
        limit_price: Trigger/limit price (for limit and stop orders).
        timestamp: Time the order was placed.
    """

    symbol: str
    side: OrderSide
    order_type: SimOrderType
    quantity: float
    limit_price: float = 0.0
    timestamp: datetime | None = None


@dataclass
class SimulatedFill:
    """Record of an executed (or partially executed) simulated order.

    Attributes:
        symbol: Instrument ticker.
        side: Buy or sell.
        quantity: Filled quantity.
        price: Execution price after slippage.
        commission: Commission paid.
        slippage_cost: Dollar cost of slippage.
        timestamp: Execution time.
        status: Fill outcome.
        raw_price: Price before slippage adjustment.
    """

    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage_cost: float
    timestamp: datetime
    status: FillStatus
    raw_price: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total execution cost (commission + slippage)."""
        return self.commission + self.slippage_cost

    @property
    def notional(self) -> float:
        """Notional value of the fill."""
        return self.quantity * self.price

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "raw_price": self.raw_price,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
            "total_cost": self.total_cost,
            "notional": self.notional,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
        }


class TradeSimulator:
    """Simulate order execution with realistic costs.

    Args:
        commission_model: Commission/fee calculator.
        slippage_model: Slippage calculator.
        max_volume_participation: Maximum fraction of bar volume the order
            may consume before being partially filled.
        min_order_size: Minimum order size; below this the order is rejected.
    """

    def __init__(
        self,
        commission_model: CommissionModel,
        slippage_model: SlippageModel,
        max_volume_participation: float = 0.10,
        min_order_size: float = 0.01,
    ) -> None:
        self._commission = commission_model
        self._slippage = slippage_model
        self._max_vol_participation = max_volume_participation
        self._min_order_size = min_order_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        order: SimulatedOrder,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        timestamp: datetime,
    ) -> SimulatedFill:
        """Attempt to fill a simulated order against a single OHLCV bar.

        Args:
            order: The order to execute.
            bar_open: Bar open price.
            bar_high: Bar high price.
            bar_low: Bar low price.
            bar_close: Bar close price.
            bar_volume: Bar volume.
            timestamp: Bar timestamp.

        Returns:
            A :class:`SimulatedFill` describing the outcome.
        """
        if order.quantity < self._min_order_size:
            logger.debug("Order rejected: quantity {:.4f} < min {:.4f}",
                         order.quantity, self._min_order_size)
            return self._reject(order, timestamp)

        if order.order_type == SimOrderType.MARKET:
            return self._fill_market(order, bar_open, bar_close, bar_volume, timestamp)

        if order.order_type == SimOrderType.LIMIT:
            return self._fill_limit(order, bar_high, bar_low, bar_close,
                                    bar_volume, timestamp)

        if order.order_type == SimOrderType.STOP:
            return self._fill_stop(order, bar_high, bar_low, bar_close,
                                   bar_volume, timestamp)

        return self._reject(order, timestamp)

    # ------------------------------------------------------------------
    # Order-type handlers
    # ------------------------------------------------------------------

    def _fill_market(
        self,
        order: SimulatedOrder,
        bar_open: float,
        bar_close: float,
        bar_volume: float,
        timestamp: datetime,
    ) -> SimulatedFill:
        """Fill a market order at bar open with slippage."""
        ref_price = bar_open
        is_buy = order.side == OrderSide.BUY
        fill_qty = self._cap_quantity(order.quantity, bar_volume)

        exec_price = self._slippage.calculate(
            ref_price, fill_qty, bar_volume=bar_volume, is_buy=is_buy,
        )
        slippage_cost = abs(exec_price - ref_price) * fill_qty
        commission = self._commission.calculate(fill_qty, exec_price, is_short=not is_buy)

        status = FillStatus.FILLED if fill_qty >= order.quantity else FillStatus.PARTIAL

        logger.debug(
            "Market {} {:.4f} {} @ {:.4f} (slip={:.4f}, comm={:.4f})",
            order.side.value, fill_qty, order.symbol, exec_price,
            slippage_cost, commission,
        )

        return SimulatedFill(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=exec_price,
            raw_price=ref_price,
            commission=commission,
            slippage_cost=slippage_cost,
            timestamp=timestamp,
            status=status,
        )

    def _fill_limit(
        self,
        order: SimulatedOrder,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        timestamp: datetime,
    ) -> SimulatedFill:
        """Fill a limit order if the price trades through the limit."""
        is_buy = order.side == OrderSide.BUY
        triggered = (bar_low <= order.limit_price) if is_buy else (bar_high >= order.limit_price)

        if not triggered:
            return SimulatedFill(
                symbol=order.symbol,
                side=order.side,
                quantity=0.0,
                price=0.0,
                raw_price=order.limit_price,
                commission=0.0,
                slippage_cost=0.0,
                timestamp=timestamp,
                status=FillStatus.PENDING,
            )

        ref_price = order.limit_price
        fill_qty = self._cap_quantity(order.quantity, bar_volume)
        exec_price = self._slippage.calculate(
            ref_price, fill_qty, bar_volume=bar_volume, is_buy=is_buy,
        )
        slippage_cost = abs(exec_price - ref_price) * fill_qty
        commission = self._commission.calculate(fill_qty, exec_price, is_short=not is_buy)
        status = FillStatus.FILLED if fill_qty >= order.quantity else FillStatus.PARTIAL

        return SimulatedFill(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=exec_price,
            raw_price=ref_price,
            commission=commission,
            slippage_cost=slippage_cost,
            timestamp=timestamp,
            status=status,
        )

    def _fill_stop(
        self,
        order: SimulatedOrder,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        timestamp: datetime,
    ) -> SimulatedFill:
        """Fill a stop order if the stop price is breached."""
        is_buy = order.side == OrderSide.BUY
        triggered = (bar_high >= order.limit_price) if is_buy else (bar_low <= order.limit_price)

        if not triggered:
            return SimulatedFill(
                symbol=order.symbol,
                side=order.side,
                quantity=0.0,
                price=0.0,
                raw_price=order.limit_price,
                commission=0.0,
                slippage_cost=0.0,
                timestamp=timestamp,
                status=FillStatus.PENDING,
            )

        ref_price = order.limit_price
        fill_qty = self._cap_quantity(order.quantity, bar_volume)
        exec_price = self._slippage.calculate(
            ref_price, fill_qty, bar_volume=bar_volume, is_buy=is_buy,
        )
        slippage_cost = abs(exec_price - ref_price) * fill_qty
        commission = self._commission.calculate(fill_qty, exec_price, is_short=not is_buy)
        status = FillStatus.FILLED if fill_qty >= order.quantity else FillStatus.PARTIAL

        return SimulatedFill(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=exec_price,
            raw_price=ref_price,
            commission=commission,
            slippage_cost=slippage_cost,
            timestamp=timestamp,
            status=status,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cap_quantity(self, desired: float, bar_volume: float) -> float:
        """Cap fill quantity to a fraction of bar volume."""
        if bar_volume <= 0:
            return desired
        max_fill = bar_volume * self._max_vol_participation
        return min(desired, max_fill) if max_fill > 0 else desired

    def _reject(self, order: SimulatedOrder, timestamp: datetime) -> SimulatedFill:
        """Return a rejected fill."""
        return SimulatedFill(
            symbol=order.symbol,
            side=order.side,
            quantity=0.0,
            price=0.0,
            commission=0.0,
            slippage_cost=0.0,
            timestamp=timestamp,
            status=FillStatus.REJECTED,
        )
