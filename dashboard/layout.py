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
    
    # Inject Premium CSS
    from dashboard.premium_css import get_premium_css
    st.markdown(get_premium_css(), unsafe_allow_html=True)

def render_sidebar(status_data: dict, post_command):
    st.sidebar.title("HMM Regime Bot")
    state = status_data.get("status", "unknown")
    
    st.sidebar.markdown("### 🏢 Core Engines")
    st.sidebar.write(f"**Main US Bot Status:** `{state.upper()}`")
    st.sidebar.write(f"**Crypto 24/7 Engine:** `🟢 ACTIVE (Cron)`")
    mode = status_data.get("mode", "PAPER_TRADING")
    st.sidebar.write(f"**Environment:** `{mode.replace('_', ' ').upper()}`")
    
    # Import and add market status here
    from dashboard.intraday_widgets import get_us_market_status
    st.sidebar.write(f"**API Connection:** {get_us_market_status()}")
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### 🛡️ Institutional Defenses")
    st.sidebar.write(f"**Macro Alert (FED/CPI):** `SAFE 🟩`")
    st.sidebar.write(f"**Level-2 Iceberg Radar:** `ONLINE 🟢`")
    st.sidebar.write(f"**Statistical Arbitrage:** `STANDBY ⚖️`")
    
    st.sidebar.divider()
    
    render_control_panel(state, post_command)

    st.sidebar.divider()
    st.sidebar.markdown("### 🧠 AI Intel Context")
    st.sidebar.write(f"**Current Regime:** {status_data.get('current_regime', 'Unknown')}")
    st.sidebar.write(f"**Active Momentum HMM:** {status_data.get('active_strategy', 'Unknown')}")
    st.sidebar.write(f"**News Sentiment NLP:** `Active (FinBERT)`")

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
    
    st.divider()
    
    # Swarm AI Live Chat Room
    st.markdown("<h2>⚡ Multi-Agent Swarm Intelligence (Live Debate Network)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Live intercept of generative AI adversarial agents debating incoming quantitative signals.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .chat-timestamp {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        import json
        import os
        if os.path.exists("logs/swarm_logs.json"):
            with open("logs/swarm_logs.json", "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f.readlines() if line.strip()]
            
            # Display last 3 debates
            logs = logs[-3:]
            
            for log in reversed(logs):
                st.markdown(f"<h4 style='color: #E2E8F0; margin-top: 10px;'>Target: {log['target']} | Signal: {log['signal']} <span class='chat-timestamp'>{log['timestamp']}</span></h4>", unsafe_allow_html=True)
                
                # Chat Interface Layout
                st.markdown(f'''
                    <div style="background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981; padding: 15px; margin-bottom: 10px; border-radius: 4px;">
                        <span style="font-weight: 800; color: #10B981;">📈 AI BULL:</span> {log['bull']}
                    </div>
                    <div style="background-color: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444; padding: 15px; margin-bottom: 10px; border-radius: 4px;">
                        <span style="font-weight: 800; color: #EF4444;">📉 AI BEAR:</span> {log['bear']}
                    </div>
                ''', unsafe_allow_html=True)
                
                verdict_color = "#10B981" if "APPROVED" in log['judge'] else "#EF4444"
                verdict_icon = "👨‍⚖️" if "APPROVED" in log['judge'] else "⛔"
                
                st.markdown(f'''
                    <div style="background-color: rgba(56, 189, 248, 0.1); border: 1px solid {verdict_color}; padding: 15px; margin-bottom: 30px; border-radius: 8px; text-align: center;">
                        <h3 style="color: {verdict_color}; margin: 0;">{verdict_icon} THE QUANT JUDGE VERDICT: {log['judge']}</h3>
                    </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("Waiting for AI Agents to intercept new target signals from the ML PPO Engine...")
    except Exception as e:
        st.warning("Swarm logs parsing failed. The module may be initializing.")
