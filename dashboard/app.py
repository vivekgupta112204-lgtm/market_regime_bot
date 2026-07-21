"""Streamlit Application Entry Point."""

import time
import streamlit as st
import threading
import sys
from pathlib import Path

# Add project root to path so relative imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.layout import configure_page, render_sidebar, render_main_content
from dashboard.live_data import fetch_status, fetch_portfolio, fetch_positions, fetch_performance, fetch_history, post_command

def main():
    configure_page()
    
    st_autorefresh = True

    status_data = fetch_status()
    portfolio = fetch_portfolio()
    positions = fetch_positions()
    perf = fetch_performance()
    hist = fetch_history()

    render_sidebar(status_data, post_command)
    
    if not status_data:
        st.error("Cannot connect to API Server. Is the backend running?")
        st.stop()

    render_main_content(portfolio, perf, positions, hist)

    if st_autorefresh:
        time.sleep(2)
        st.rerun()

def start_dashboard():
    """Programmatic entry point for the master runner script."""
    import subprocess, sys
    st_file = str(Path(__file__).resolve())
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", st_file, "--server.port", "8501", "--server.headless", "true"])

if __name__ == "__main__":
    main()
