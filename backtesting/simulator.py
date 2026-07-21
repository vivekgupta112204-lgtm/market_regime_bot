"""Portfolio simulator for backtesting.

Tracks cash, portfolio value, margin, open/closed positions, and daily
equity snapshots throughout a backtest run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

import pandas as pd
from loguru import logger

from backtesting.trade_simulator import SimulatedFill, OrderSide, FillStatus


@dataclass
class SimPosition:
    """An open position tracked during simulation.

    Attributes:
        symbol: Instrument ticker.
        direction: ``'LONG'`` or ``'SHORT'``.
        entry_price: Weighted-average entry price.
        quantity: Current position size.
        stop_loss: Stop-loss price level.
        take_profit: Take-profit price level.
        opened_at: Timestamp when the position was first opened.
        commission_paid: Cumulative commission paid on this position.
        slippage_paid: Cumulative slippage cost on this position.
    """

    symbol: str
    direction: str
    entry_price: float
    quantity: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    opened_at: datetime | None = None
    commission_paid: float = 0.0
    slippage_paid: float = 0.0

    @property
    def notional(self) -> float:
        """Notional value at entry."""
        return self.entry_price * self.quantity


@dataclass
class ClosedTrade:
    """Record of a completed round-trip trade.

    Attributes:
        symbol: Instrument ticker.
        direction: ``'LONG'`` or ``'SHORT'``.
        entry_price: Average entry price.
        exit_price: Average exit price.
        quantity: Number of units traded.
        pnl: Realised profit/loss (after costs).
        pnl_pct: Percentage return on the trade.
        commission: Total commission paid (entry + exit).
        slippage: Total slippage cost (entry + exit).
        entry_time: When the position was opened.
        exit_time: When the position was closed.
        bars_held: Number of bars the position was held.
    """

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    bars_held: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "commission": round(self.commission, 4),
            "slippage": round(self.slippage, 4),
            "entry_time": self.entry_time.isoformat() if self.entry_time else "",
            "exit_time": self.exit_time.isoformat() if self.exit_time else "",
            "bars_held": self.bars_held,
        }


@dataclass
class EquitySnapshot:
    """Point-in-time portfolio snapshot.

    Attributes:
        timestamp: Snapshot time.
        cash: Available cash.
        portfolio_value: Total portfolio value (cash + positions MTM).
        open_positions: Number of open positions.
        daily_pnl: Profit/loss since the previous snapshot.
    """

    timestamp: datetime
    cash: float
    portfolio_value: float
    open_positions: int
    daily_pnl: float = 0.0


class PortfolioSimulator:
    """Simulate portfolio state throughout a backtest.

    Args:
        initial_capital: Starting cash balance.
        symbol: The instrument being traded.
    """

    def __init__(self, initial_capital: float, symbol: str = "") -> None:
        self._initial_capital = initial_capital
        self._symbol = symbol
        self._cash = initial_capital
        self._positions: dict[str, SimPosition] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._equity_curve: list[EquitySnapshot] = []
        self._bar_counter: dict[str, int] = {}
        self._high_water_mark = initial_capital

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def initial_capital(self) -> float:
        """Starting capital."""
        return self._initial_capital

    @property
    def positions(self) -> dict[str, SimPosition]:
        """Currently open positions."""
        return dict(self._positions)

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        """All completed round-trip trades."""
        return list(self._closed_trades)

    @property
    def equity_snapshots(self) -> list[EquitySnapshot]:
        """Historical equity snapshots."""
        return list(self._equity_curve)

    @property
    def portfolio_value(self) -> float:
        """Current total portfolio value."""
        return self._cash + sum(
            p.quantity * p.entry_price for p in self._positions.values()
        )

    @property
    def high_water_mark(self) -> float:
        """Highest portfolio value achieved."""
        return self._high_water_mark

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_fill(
        self,
        fill: SimulatedFill,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> None:
        """Process a simulated fill and update portfolio state.

        Args:
            fill: The execution fill from :class:`TradeSimulator`.
            stop_loss: Stop-loss price for the new position.
            take_profit: Take-profit price for the new position.
        """
        if fill.status in (FillStatus.REJECTED, FillStatus.PENDING):
            return

        is_buy = fill.side == OrderSide.BUY
        symbol = fill.symbol

        existing = self._positions.get(symbol)

        if existing:
            is_closing = (
                (existing.direction == "LONG" and not is_buy) or
                (existing.direction == "SHORT" and is_buy)
            )
            if is_closing:
                self._close_position(existing, fill)
                return

        self._open_position(fill, stop_loss, take_profit)

    def check_stops(
        self,
        symbol: str,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        timestamp: datetime,
    ) -> ClosedTrade | None:
        """Check if stop-loss or take-profit has been hit.

        Args:
            symbol: Instrument to check.
            bar_high: Current bar high.
            bar_low: Current bar low.
            bar_close: Current bar close.
            timestamp: Current bar timestamp.

        Returns:
            A :class:`ClosedTrade` if a stop was triggered, else ``None``.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return None

        self._bar_counter[symbol] = self._bar_counter.get(symbol, 0) + 1

        if pos.direction == "LONG":
            if pos.stop_loss > 0 and bar_low <= pos.stop_loss:
                return self._force_close(pos, pos.stop_loss, timestamp, "stop_loss")
            if pos.take_profit > 0 and bar_high >= pos.take_profit:
                return self._force_close(pos, pos.take_profit, timestamp, "take_profit")
        else:
            if pos.stop_loss > 0 and bar_high >= pos.stop_loss:
                return self._force_close(pos, pos.stop_loss, timestamp, "stop_loss")
            if pos.take_profit > 0 and bar_low <= pos.take_profit:
                return self._force_close(pos, pos.take_profit, timestamp, "take_profit")

        return None

    def snapshot(self, timestamp: datetime, current_price: float) -> EquitySnapshot:
        """Record an equity snapshot at the current bar.

        Args:
            timestamp: Current bar timestamp.
            current_price: Current market price for MTM.

        Returns:
            The newly created :class:`EquitySnapshot`.
        """
        mtm_value = self._cash
        for pos in self._positions.values():
            if pos.direction == "LONG":
                mtm_value += pos.quantity * current_price
            else:
                mtm_value += pos.quantity * (2 * pos.entry_price - current_price)

        prev_value = self._equity_curve[-1].portfolio_value if self._equity_curve else self._initial_capital
        daily_pnl = mtm_value - prev_value

        if mtm_value > self._high_water_mark:
            self._high_water_mark = mtm_value

        snap = EquitySnapshot(
            timestamp=timestamp,
            cash=self._cash,
            portfolio_value=mtm_value,
            open_positions=len(self._positions),
            daily_pnl=daily_pnl,
        )
        self._equity_curve.append(snap)
        return snap

    def get_equity_series(self) -> pd.Series:
        """Return portfolio value as a pandas Series indexed by timestamp."""
        if not self._equity_curve:
            return pd.Series(dtype=float)
        return pd.Series(
            {s.timestamp: s.portfolio_value for s in self._equity_curve},
            name="portfolio_value",
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the portfolio state."""
        return {
            "initial_capital": self._initial_capital,
            "final_capital": round(self.portfolio_value, 2),
            "cash": round(self._cash, 2),
            "open_positions": len(self._positions),
            "closed_trades": len(self._closed_trades),
            "high_water_mark": round(self._high_water_mark, 2),
        }

    # ------------------------------------------------------------------
    # Internal: Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        fill: SimulatedFill,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Open a new position or add to an existing one."""
        direction = "LONG" if fill.side == OrderSide.BUY else "SHORT"
        cost = fill.quantity * fill.price + fill.commission

        if direction == "LONG":
            self._cash -= cost
        else:
            self._cash -= fill.commission
            self._cash += fill.quantity * fill.price

        existing = self._positions.get(fill.symbol)
        if existing and existing.direction == direction:
            total_qty = existing.quantity + fill.quantity
            avg_price = (
                (existing.entry_price * existing.quantity + fill.price * fill.quantity)
                / total_qty
            )
            existing.entry_price = avg_price
            existing.quantity = total_qty
            existing.commission_paid += fill.commission
            existing.slippage_paid += fill.slippage_cost
        else:
            self._positions[fill.symbol] = SimPosition(
                symbol=fill.symbol,
                direction=direction,
                entry_price=fill.price,
                quantity=fill.quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=fill.timestamp,
                commission_paid=fill.commission,
                slippage_paid=fill.slippage_cost,
            )
            self._bar_counter[fill.symbol] = 0

        logger.debug(
            "Opened {} {} {:.4f} @ {:.4f}",
            direction, fill.symbol, fill.quantity, fill.price,
        )

    def _close_position(self, pos: SimPosition, fill: SimulatedFill) -> None:
        """Close an existing position with the given fill."""
        close_qty = min(fill.quantity, pos.quantity)

        if pos.direction == "LONG":
            gross_pnl = (fill.price - pos.entry_price) * close_qty
            self._cash += close_qty * fill.price - fill.commission
        else:
            gross_pnl = (pos.entry_price - fill.price) * close_qty
            self._cash -= close_qty * fill.price
            self._cash -= fill.commission
            self._cash += close_qty * pos.entry_price + gross_pnl

        total_commission = pos.commission_paid + fill.commission
        total_slippage = pos.slippage_paid + fill.slippage_cost
        net_pnl = gross_pnl - total_commission - total_slippage
        pnl_pct = net_pnl / (pos.entry_price * close_qty) if pos.entry_price > 0 else 0.0
        bars_held = self._bar_counter.get(pos.symbol, 0)

        trade = ClosedTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill.price,
            quantity=close_qty,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            slippage=total_slippage,
            entry_time=pos.opened_at,
            exit_time=fill.timestamp,
            bars_held=bars_held,
        )
        self._closed_trades.append(trade)

        remaining = pos.quantity - close_qty
        if remaining > 0:
            pos.quantity = remaining
        else:
            del self._positions[pos.symbol]
            self._bar_counter.pop(pos.symbol, None)

        logger.debug(
            "Closed {} {} {:.4f} @ {:.4f}, PnL={:.4f}",
            pos.direction, pos.symbol, close_qty, fill.price, net_pnl,
        )

    def _force_close(
        self,
        pos: SimPosition,
        exit_price: float,
        timestamp: datetime,
        reason: str,
    ) -> ClosedTrade:
        """Force-close a position at a given price (for SL/TP hits)."""
        if pos.direction == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            self._cash += pos.quantity * exit_price
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            self._cash += pos.quantity * pos.entry_price + gross_pnl

        total_commission = pos.commission_paid
        total_slippage = pos.slippage_paid
        net_pnl = gross_pnl - total_commission - total_slippage
        pnl_pct = net_pnl / (pos.entry_price * pos.quantity) if pos.entry_price > 0 else 0.0
        bars_held = self._bar_counter.get(pos.symbol, 0)

        trade = ClosedTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            slippage=total_slippage,
            entry_time=pos.opened_at,
            exit_time=timestamp,
            bars_held=bars_held,
        )
        self._closed_trades.append(trade)

        del self._positions[pos.symbol]
        self._bar_counter.pop(pos.symbol, None)

        logger.info(
            "{} triggered for {} @ {:.4f}, PnL={:.4f}",
            reason.upper(), pos.symbol, exit_price, net_pnl,
        )
        return trade
