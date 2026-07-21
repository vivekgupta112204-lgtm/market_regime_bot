"""Plotly chart generators for backtesting reports.

Generates equity curve, drawdown, portfolio value, trade distribution,
monthly/yearly returns heatmaps, regime timeline, PnL distribution,
rolling Sharpe, and rolling drawdown charts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from analytics.equity_curve import EquityCurve
from analytics.drawdown import DrawdownAnalyzer


# ── Colour palette ────────────────────────────────────────────────────
_PALETTE = {
    "primary": "#6366F1",
    "secondary": "#22D3EE",
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral": "#94A3B8",
    "background": "#0F172A",
    "paper": "#1E293B",
    "grid": "#334155",
    "text": "#E2E8F0",
}

_REGIME_COLORS = {
    "Bull Market": "#10B981",
    "Bear Market": "#EF4444",
    "Sideways Market": "#F59E0B",
    "High Volatility": "#8B5CF6",
    "Low Volatility": "#06B6D4",
}


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply a dark professional theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PALETTE["paper"],
        plot_bgcolor=_PALETTE["background"],
        font=dict(family="Inter, sans-serif", color=_PALETTE["text"]),
        xaxis=dict(gridcolor=_PALETTE["grid"], showgrid=True),
        yaxis=dict(gridcolor=_PALETTE["grid"], showgrid=True),
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def equity_curve_chart(
    equity_series: pd.Series,
    benchmark_series: pd.Series | None = None,
    title: str = "Equity Curve",
) -> go.Figure:
    """Generate an equity curve chart.

    Args:
        equity_series: Portfolio value series.
        benchmark_series: Optional benchmark for overlay.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_series.index, y=equity_series.values,
        name="Strategy", line=dict(color=_PALETTE["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.1)",
    ))

    if benchmark_series is not None and len(benchmark_series) > 0:
        fig.add_trace(go.Scatter(
            x=benchmark_series.index, y=benchmark_series.values,
            name="Benchmark", line=dict(color=_PALETTE["neutral"], width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        legend=dict(x=0.02, y=0.98),
    )
    return _apply_theme(fig)


def drawdown_chart(equity_series: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Generate a drawdown chart.

    Args:
        equity_series: Portfolio value series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    analyzer = DrawdownAnalyzer(equity_series)
    dd = -analyzer.drawdown_series

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown %", line=dict(color=_PALETTE["negative"], width=1.5),
        fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.2)",
    ))

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown (%)",
    )
    return _apply_theme(fig)


def portfolio_value_chart(equity_series: pd.Series, title: str = "Portfolio Value") -> go.Figure:
    """Generate a portfolio value area chart.

    Args:
        equity_series: Portfolio value series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    curve = EquityCurve(equity_series)
    cum = curve.cumulative_returns

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        name="Cumulative Return %",
        line=dict(color=_PALETTE["secondary"], width=2),
        fill="tozeroy", fillcolor="rgba(34, 211, 238, 0.1)",
    ))

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Cumulative Return (%)",
    )
    return _apply_theme(fig)


def trade_distribution_chart(trades: list[dict[str, Any]], title: str = "Trade Distribution") -> go.Figure:
    """Generate a trade PnL distribution histogram.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if not trades:
        fig = go.Figure()
        fig.update_layout(title=title)
        return _apply_theme(fig)

    pnls = [t["pnl"] for t in trades]
    colors = [_PALETTE["positive"] if p > 0 else _PALETTE["negative"] for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnls, nbinsx=30, name="PnL",
        marker=dict(color=_PALETTE["primary"], line=dict(width=1, color=_PALETTE["text"])),
    ))

    fig.update_layout(
        title=title, xaxis_title="PnL ($)", yaxis_title="Frequency",
    )
    return _apply_theme(fig)


def monthly_returns_chart(equity_series: pd.Series, title: str = "Monthly Returns") -> go.Figure:
    """Generate a monthly returns heatmap.

    Args:
        equity_series: Portfolio value series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    curve = EquityCurve(equity_series)
    monthly = curve.monthly_returns()

    if monthly.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return _apply_theme(fig)

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=monthly.columns.tolist(),
        y=[str(y) for y in monthly.index],
        colorscale=[
            [0.0, _PALETTE["negative"]],
            [0.5, _PALETTE["paper"]],
            [1.0, _PALETTE["positive"]],
        ],
        zmid=0, text=np.round(monthly.values, 1), texttemplate="%{text}%",
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title, xaxis_title="Month", yaxis_title="Year",
    )
    return _apply_theme(fig)


def yearly_returns_chart(equity_series: pd.Series, title: str = "Yearly Returns") -> go.Figure:
    """Generate a yearly returns bar chart.

    Args:
        equity_series: Portfolio value series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    curve = EquityCurve(equity_series)
    yearly = curve.yearly_returns()

    if yearly.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return _apply_theme(fig)

    colors = [_PALETTE["positive"] if v > 0 else _PALETTE["negative"] for v in yearly.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(d.year) if hasattr(d, 'year') else str(d) for d in yearly.index],
        y=yearly.values,
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in yearly.values],
        textposition="outside",
    ))

    fig.update_layout(
        title=title, xaxis_title="Year", yaxis_title="Return (%)",
    )
    return _apply_theme(fig)


def regime_timeline_chart(
    equity_series: pd.Series,
    regimes: list[dict[str, Any]] | list[str],
    title: str = "Regime Timeline",
) -> go.Figure:
    """Generate a regime timeline overlay on the equity curve.

    Args:
        equity_series: Portfolio value series.
        regimes: List of regime names or dicts with ``regime`` key.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_series.index, y=equity_series.values,
        name="Portfolio", line=dict(color=_PALETTE["primary"], width=2),
    ))

    regime_names = [
        r if isinstance(r, str) else r.get("regime", "Unknown")
        for r in regimes
    ]

    if len(regime_names) == len(equity_series):
        idx = list(equity_series.index)
        i = 0
        while i < len(regime_names):
            current = regime_names[i]
            start_idx = i
            while i < len(regime_names) and regime_names[i] == current:
                i += 1
            end_idx = i - 1

            color = _REGIME_COLORS.get(current, _PALETTE["neutral"])
            fig.add_vrect(
                x0=idx[start_idx], x1=idx[min(end_idx, len(idx) - 1)],
                fillcolor=color, opacity=0.15,
                layer="below", line_width=0,
                annotation_text=current if (end_idx - start_idx) > 10 else "",
                annotation_position="top left",
                annotation_font_size=9,
            )

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    )
    return _apply_theme(fig)


def pnl_distribution_chart(trades: list[dict[str, Any]], title: str = "PnL Distribution") -> go.Figure:
    """Generate a PnL percentage distribution chart.

    Args:
        trades: List of trade dictionaries with ``pnl_pct`` key.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if not trades:
        fig = go.Figure()
        fig.update_layout(title=title)
        return _apply_theme(fig)

    pnl_pcts = [t.get("pnl_pct", 0.0) * 100.0 for t in trades]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl_pcts, nbinsx=30, name="Return %",
        marker=dict(color=_PALETTE["secondary"]),
    ))

    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color=_PALETTE["text"])
    fig.add_vline(
        x=float(np.mean(pnl_pcts)), line_width=1.5,
        line_dash="dot", line_color=_PALETTE["positive"],
        annotation_text=f"Mean: {np.mean(pnl_pcts):.2f}%",
    )

    fig.update_layout(
        title=title, xaxis_title="Return (%)", yaxis_title="Frequency",
    )
    return _apply_theme(fig)


def rolling_sharpe_chart(
    equity_series: pd.Series,
    window: int = 63,
    title: str = "Rolling Sharpe Ratio",
) -> go.Figure:
    """Generate a rolling Sharpe ratio chart.

    Args:
        equity_series: Portfolio value series.
        window: Rolling window size.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    curve = EquityCurve(equity_series)
    rolling = curve.rolling_sharpe(window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        name=f"Rolling Sharpe ({window}d)",
        line=dict(color=_PALETTE["primary"], width=1.5),
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=_PALETTE["neutral"])
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=_PALETTE["positive"],
                  annotation_text="Sharpe = 1.0")

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Sharpe Ratio",
    )
    return _apply_theme(fig)


def rolling_drawdown_chart(
    equity_series: pd.Series,
    title: str = "Rolling Drawdown",
) -> go.Figure:
    """Generate a rolling drawdown chart.

    Args:
        equity_series: Portfolio value series.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    return drawdown_chart(equity_series, title)


def generate_all_charts(
    equity_series: pd.Series,
    trades: list[dict[str, Any]],
    regimes: list[str] | None = None,
    benchmark_series: pd.Series | None = None,
) -> dict[str, go.Figure]:
    """Generate all available charts.

    Args:
        equity_series: Portfolio value series.
        trades: List of trade dictionaries.
        regimes: Optional regime labels.
        benchmark_series: Optional benchmark series.

    Returns:
        Dictionary mapping chart names to Plotly figures.
    """
    charts: dict[str, go.Figure] = {
        "equity_curve": equity_curve_chart(equity_series, benchmark_series),
        "drawdown": drawdown_chart(equity_series),
        "portfolio_value": portfolio_value_chart(equity_series),
        "trade_distribution": trade_distribution_chart(trades),
        "monthly_returns": monthly_returns_chart(equity_series),
        "yearly_returns": yearly_returns_chart(equity_series),
        "pnl_distribution": pnl_distribution_chart(trades),
        "rolling_sharpe": rolling_sharpe_chart(equity_series),
        "rolling_drawdown": rolling_drawdown_chart(equity_series),
    }

    if regimes:
        charts["regime_timeline"] = regime_timeline_chart(equity_series, regimes)

    logger.info("Generated {} charts", len(charts))
    return charts
