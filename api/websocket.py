"""WebSocket Connection Manager for live dashboard streaming."""

from __future__ import annotations
import json
from typing import List
from fastapi import WebSocket
from loguru import logger

class ConnectionManager:
    """Manages active websocket connections and broadcasts events."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected.")

    async def broadcast(self, event_type: str, data: dict):
        """Broadcast JSON payload to all connected clients."""
        if not self.active_connections:
            return

        payload = json.dumps({"event": event_type, "data": data})
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                self.disconnect(connection)

# Singleton instance exported for use everywhere
ws_manager = ConnectionManager()
