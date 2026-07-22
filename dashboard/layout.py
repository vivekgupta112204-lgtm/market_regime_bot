"""Main page layout structuring."""

import streamlit as st
from dashboard.widgets import render_metric_card, render_positions_table, render_control_panel
from dashboard.charts import create_equity_curve_chart, create_drawdown_chart

def configure_page():
    st.set_page_config(
        page_title="HMM Trade Bot | Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar(status_data: dict, post_command):
    st.sidebar.title("HMM Regime Bot")
    state = status_data.get("status", "unknown")
    st.sidebar.write(f"**Status:** `{state.upper()}`")
    mode = status_data.get("mode", "PAPER_TRADING")
    st.sidebar.write(f"**Mode:** `{mode.replace('_', ' ').upper()}`")
    
    # Import and add market status here
    from dashboard.intraday_widgets import get_us_market_status
    st.sidebar.write(f"{get_us_market_status()}")
    
    st.sidebar.divider()
    render_control_panel(state, post_command)

    st.sidebar.divider()
    st.sidebar.write("### AI Context")
    st.sidebar.write(f"**Current Regime:** {status_data.get('current_regime', 'Unknown')}")
    st.sidebar.write(f"**Active Strategy:** {status_data.get('active_strategy', 'Unknown')}")

def render_main_content(portfolio: dict, performance: dict, positions: list, history: dict):
    # Top Row Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Portfolio Value", f"${portfolio.get('portfolio_value', 0):,.2f}")
    with c2:
        render_metric_card("Buying Power", f"${portfolio.get('buying_power', 0):,.2f}")
    with c3:
        render_metric_card("Daily PnL", f"${performance.get('daily_pnl', 0):,.2f}", delta=f"{performance.get('daily_pnl', 0):.2f}")
    with c4:
        render_metric_card("Total Return", f"{performance.get('total_return_pct', 0):.2f}%")

    st.divider()

    # Second Row: Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(create_equity_curve_chart(history.get("equity")), use_container_width=True)
    with col_chart2:
        # Pass equity data again for drawdown calculation natively
        st.plotly_chart(create_drawdown_chart(history.get("equity")), use_container_width=True)

    st.divider()

    # Active Positions
    st.subheader("Active Positions")
    render_positions_table(positions)
