"""Client data fetching for dashboard UI."""

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
API_KEY = "secret_dashboard_key_123"

def _get(endpoint: str) -> dict:
    """Helper to fetch from the local API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}", 
            headers={"X-API-Key": API_KEY},
            timeout=2.0
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=1)  # Cache for 1 second to avoid spamming the backend during rerenders
def fetch_status() -> dict:
    return _get("/status")

@st.cache_data(ttl=1)
def fetch_portfolio() -> dict:
    return _get("/portfolio")

@st.cache_data(ttl=1)
def fetch_positions() -> list:
    return _get("/positions") or []

@st.cache_data(ttl=5)
def fetch_performance() -> dict:
    return _get("/performance")
    
@st.cache_data(ttl=5)
def fetch_history() -> dict:
    return _get("/history")

@st.cache_data(ttl=2)
def fetch_health() -> dict:
    return _get("/health")

def post_command(command: str) -> bool:
    """Sends a control command (start/stop) to the bot."""
    try:
        response = requests.post(f"{API_BASE_URL}/{command}", headers={"X-API-Key": API_KEY})
        return response.status_code == 200
    except Exception:
        return False
