"""Plotly integrations specifically formatted for Streamlit."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

def create_equity_curve_chart(equity_data: list[float] | None = None) -> go.Figure:
    """Renders a simple blue equity curve plot."""
    if not equity_data or len(equity_data) == 0:
        # Mock data if not connected properly just for visual placeholder
        equity_data = np.cumsum(np.random.normal(100, 200, 100)) + 100000

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_data,
        mode='lines',
        line=dict(color='#00F0FF', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.1)',
        name='Equity'
    ))
    fig.update_layout(
        title="Live Equity Curve",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)"
    )
    return fig

def create_drawdown_chart(equity_data: list[float] | None = None) -> go.Figure:
    """Renders the drawdown chart with red fill."""
    if not equity_data or len(equity_data) == 0:
        drawdown_data = -np.abs(np.random.normal(0, 2, 100))
    else:
        eq = np.array(equity_data)
        running_max = np.maximum.accumulate(eq)
        drawdown_data = ((eq - running_max) / running_max) * 100
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=drawdown_data,
        mode='lines',
        line=dict(color='#FF3A3A', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 58, 58, 0.2)',
        name='Drawdown'
    ))
    fig.update_layout(
        title="Current Drawdown (%)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed")
    )
    return fig
