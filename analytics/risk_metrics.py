"""Risk metrics for portfolio analytics.

Computes Value at Risk, Conditional VaR, volatility, Beta, Alpha,
Information Ratio, and Ulcer Index.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

TRADING_DAYS_PER_YEAR = 252


def portfolio_volatility(
    equity_series: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualised portfolio volatility.

    Args:
        equity_series: Time-indexed portfolio value series.
        trading_days: Trading days per year.

    Returns:
        Annualised volatility as a percentage.
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days) * 100.0)


def value_at_risk(
    equity_series: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk.

    Args:
        equity_series: Time-indexed portfolio value series.
        confidence: Confidence level (e.g. 0.95 for 95%).
        method: ``'historical'`` or ``'parametric'``.

    Returns:
        VaR as a positive percentage (daily).
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0

    if method == "parametric":
        from scipy.stats import norm
        mean = returns.mean()
        std = returns.std()
        z = norm.ppf(1 - confidence)
        var = -(mean + z * std) * 100.0
    else:
        percentile = (1 - confidence) * 100.0
        var = -float(np.percentile(returns, percentile)) * 100.0

    return max(var, 0.0)


def conditional_var(
    equity_series: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        equity_series: Time-indexed portfolio value series.
        confidence: Confidence level.

    Returns:
        CVaR as a positive percentage (daily).
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    percentile = (1 - confidence) * 100.0
    threshold = np.percentile(returns, percentile)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return -float(tail.mean()) * 100.0


def beta(
    equity_series: pd.Series,
    benchmark_series: pd.Series,
) -> float:
    """Calculate portfolio Beta relative to a benchmark.

    Args:
        equity_series: Portfolio value series.
        benchmark_series: Benchmark value series.

    Returns:
        Beta coefficient.
    """
    port_returns = equity_series.pct_change().dropna()
    bench_returns = benchmark_series.pct_change().dropna()

    aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
    if len(aligned) < 2:
        return 0.0

    cov = np.cov(aligned["port"], aligned["bench"])
    var_bench = cov[1, 1]
    if var_bench == 0:
        return 0.0
    return float(cov[0, 1] / var_bench)


def alpha(
    equity_series: pd.Series,
    benchmark_series: pd.Series,
    risk_free: float = 0.0,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Jensen's Alpha.

    Args:
        equity_series: Portfolio value series.
        benchmark_series: Benchmark value series.
        risk_free: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Annualised alpha as a percentage.
    """
    port_returns = equity_series.pct_change().dropna()
    bench_returns = benchmark_series.pct_change().dropna()

    aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
    if len(aligned) < 2:
        return 0.0

    port_annual = float(aligned["port"].mean() * trading_days)
    bench_annual = float(aligned["bench"].mean() * trading_days)
    b = beta(equity_series, benchmark_series)
    rf = risk_free

    return (port_annual - rf - b * (bench_annual - rf)) * 100.0


def information_ratio(
    equity_series: pd.Series,
    benchmark_series: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Information Ratio.

    Args:
        equity_series: Portfolio value series.
        benchmark_series: Benchmark value series.
        trading_days: Trading days per year.

    Returns:
        Annualised Information Ratio.
    """
    port_returns = equity_series.pct_change().dropna()
    bench_returns = benchmark_series.pct_change().dropna()

    aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
    if len(aligned) < 2:
        return 0.0

    active_return = aligned["port"] - aligned["bench"]
    tracking_error = active_return.std()
    if tracking_error == 0:
        return 0.0
    return float((active_return.mean() / tracking_error) * np.sqrt(trading_days))


def ulcer_index(equity_series: pd.Series) -> float:
    """Calculate Ulcer Index (measures downside volatility).

    Args:
        equity_series: Time-indexed portfolio value series.

    Returns:
        Ulcer Index value.
    """
    if len(equity_series) < 2:
        return 0.0
    running_max = equity_series.cummax()
    pct_drawdowns = ((equity_series - running_max) / running_max) * 100.0
    squared_dd = pct_drawdowns ** 2
    return float(np.sqrt(squared_dd.mean()))


def compute_all(
    equity_series: pd.Series,
    benchmark_series: pd.Series | None = None,
    confidence: float = 0.95,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute all risk metrics.

    Args:
        equity_series: Portfolio value series.
        benchmark_series: Optional benchmark series for relative metrics.
        confidence: VaR confidence level.
        trading_days: Trading days per year.

    Returns:
        Dictionary of all risk metrics.
    """
    result: dict[str, float] = {
        "volatility": round(portfolio_volatility(equity_series, trading_days), 2),
        "var_95": round(value_at_risk(equity_series, confidence, "historical"), 4),
        "var_95_parametric": round(value_at_risk(equity_series, confidence, "parametric"), 4),
        "cvar_95": round(conditional_var(equity_series, confidence), 4),
        "ulcer_index": round(ulcer_index(equity_series), 4),
    }

    if benchmark_series is not None and len(benchmark_series) > 1:
        result["beta"] = round(beta(equity_series, benchmark_series), 4)
        result["alpha"] = round(alpha(equity_series, benchmark_series, trading_days=trading_days), 2)
        result["information_ratio"] = round(information_ratio(equity_series, benchmark_series, trading_days), 4)

    return result
