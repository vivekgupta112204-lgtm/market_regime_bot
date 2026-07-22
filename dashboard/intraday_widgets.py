"""Streamlit Intraday components."""
import streamlit as st
from datetime import datetime, time as dtime
import pytz

def get_us_market_status() -> str:
    now = datetime.now(pytz.timezone("America/New_York"))
    # Check if weekday
    if now.weekday() >= 5:
         return "🔴 CLOSED (Weekend)"
    now_time = now.time()
    if dtime(9, 30) <= now_time <= dtime(16, 0):
        return "🟢 OPEN (US Market)"
    else:
        return "🔴 CLOSED (Market Off-Hours)"

def render_intraday_header(todays_pnl: float, active_positions: int, trades_today: int, regime: str, current_strat: str, win_rate: float, remaining_risk: float, max_daily_risk: float):
    st.title("🇺🇸 US Swing & Positional AI Dashboard")
    st.markdown("---")
    
    market_status = get_us_market_status()
    st.markdown(f"**Live Market Status:** {market_status}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Today's PnL", f"${todays_pnl:,.2f}", f"{(todays_pnl/100000)*100:.2f}%")
    col2.metric("Open Positions", active_positions)
    col3.metric("Trades Executed", f"{trades_today}/5")
    col4.metric("Remaining Risk Budget", f"${remaining_risk:,.2f}", f"-{max_daily_risk}% Max Limit")

    st.markdown("### 🧠 Autonomous Swarm Status")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("HMM Current Regime", regime)
    sc2.metric("Active Strategy", current_strat)
    sc3.metric("Intraday Win Rate", f"{win_rate*100:.1f}%")
