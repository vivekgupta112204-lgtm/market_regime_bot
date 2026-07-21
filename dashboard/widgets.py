"""Reusable UI widget components for the Dashboard."""

import streamlit as st
import pandas as pd

def render_metric_card(title: str, value: str, delta: str | None = None):
    """Renders a visually appealing metric card with optional delta."""
    st.metric(label=title, value=value, delta=delta)

def render_positions_table(positions: list[dict]):
    """Renders an interactive dataframe of open positions."""
    if not positions:
        st.write("No open positions.")
        return
        
    df = pd.DataFrame(positions)
    st.dataframe(
        df,
        column_config={
            "symbol": st.column_config.TextColumn("Asset", width="medium"),
            "side": st.column_config.TextColumn("Side", width="small"),
            "qty": st.column_config.NumberColumn("Size", format="%.4f"),
            "pnl": st.column_config.NumberColumn("Unrealized PnL", format="$%.2f")
        },
        hide_index=True,
        use_container_width=True
    )

def render_control_panel(state: str, post_command_callable):
    """Renders start/stop bot buttons."""
    st.markdown("### Bot Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if state != "running":
            if st.button("▶ Start Bot", use_container_width=True):
                post_command_callable("start")
                st.rerun()
        else:
            st.button("🟢 Running", disabled=True, use_container_width=True)

    with col2:
        if state == "running":
            if st.button("⏹ Stop Bot", type="primary", use_container_width=True):
                post_command_callable("stop")
                st.rerun()
        else:
            st.button("⛔ Stopped", disabled=True, use_container_width=True)
