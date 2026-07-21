"""API routes mapping for the dashboard."""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, WebSocket
from typing import Any
from loguru import logger

from api.auth import verify_api_key
from api.websocket import ws_manager
from monitor.system_monitor import SystemMonitor

router = APIRouter()
sys_monitor = SystemMonitor()

# Global state mock variables since real state needs to integrate with the engine instance
_BOTTING_STATE = "running"
_CURRENT_REGIME = "Bull"
_ACTIVE_STRATEGY = "Trend Following"

import os

@router.get("/status")
async def get_status(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    return {
        "status": _BOTTING_STATE,
        "mode": os.environ.get("BOT_MODE", "PAPER_TRADING"),
        "current_regime": _CURRENT_REGIME,
        "active_strategy": _ACTIVE_STRATEGY
    }

@router.get("/health")
async def get_health() -> dict[str, Any]:
    # Unprotected to allow quick healthchecks
    return sys_monitor.get_system_status()
    
import os
from dotenv import load_dotenv

load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

trading_client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    try:
        from alpaca.trading.client import TradingClient
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        logger.info("Successfully bound Alpaca API client for dashboard.")
    except Exception as e:
        logger.error(f"Failed to bind Alpaca client: {e}")

@router.get("/portfolio")
async def get_portfolio(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    if trading_client:
        try:
            account = trading_client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "margin_used": 0.0,
                "open_exposure": float(account.portfolio_value) - float(account.cash),
            }
        except Exception as e:
            logger.error(f"Alpaca fetch failed: {e}")
            
    # Fallback mock
    return {
        "cash": 0.0,
        "portfolio_value": 0.0,
        "buying_power": 0.0,
        "margin_used": 0.0,
        "open_exposure": 0.0,
    }

@router.get("/positions")
async def get_positions(api_key: str = Depends(verify_api_key)) -> list[dict[str, Any]]:
    if trading_client:
        try:
            positions = trading_client.get_all_positions()
            formatted = []
            for p in positions:
                formatted.append({
                    "symbol": p.symbol,
                    "side": str(p.side),
                    "qty": float(p.qty),
                    "entry": float(p.avg_entry_price),
                    "current": float(p.current_price),
                    "pnl": float(p.unrealized_pl)
                })
            return formatted
        except Exception as e:
            logger.error(f"Alpaca pos fetch failed: {e}")
            
    return []

@router.get("/performance")
async def get_performance(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    if trading_client:
        try:
            account = trading_client.get_account()
            curr = float(account.equity)
            last = float(account.last_equity)
            daily_pnl = curr - last
            pct_change = (daily_pnl / last * 100) if last > 0 else 0
            
            return {
                "total_return_pct": pct_change,
                "daily_pnl": daily_pnl,
                "drawdown_pct": 0.0,
                "win_rate": 0.0
            }
        except Exception as e:
            logger.error(f"Failed performance config: {e}")
            
    return {
        "total_return_pct": 0.0,
        "daily_pnl": 0.0,
        "drawdown_pct": 0.0,
        "win_rate": 0.0
    }

@router.get("/history")
async def get_history(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Fetches real portfolio equity curve from Alpaca."""
    if trading_client:
        try:
            hist = trading_client.get_portfolio_history()
            return {
                "equity": hist.equity,
                "timestamps": hist.timestamp
            }
        except Exception as e:
            logger.error(f"Alpaca history error: {e}")
            
    return {"equity": [], "timestamps": []}

@router.post("/start")
async def start_bot(api_key: str = Depends(verify_api_key)):
    global _BOTTING_STATE
    _BOTTING_STATE = "running"
    await ws_manager.broadcast("status", {"state": _BOTTING_STATE})
    return {"message": "Bot starting up"}

@router.post("/stop")
async def stop_bot(api_key: str = Depends(verify_api_key)):
    global _BOTTING_STATE
    _BOTTING_STATE = "stopped"
    await ws_manager.broadcast("status", {"state": _BOTTING_STATE})
    return {"message": "Bot stopping"}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep alive and receive simple incoming client pings
            data = await websocket.receive_text()
    except Exception:
        ws_manager.disconnect(websocket)
