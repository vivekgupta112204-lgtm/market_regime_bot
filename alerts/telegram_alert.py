from __future__ import annotations

import httpx
from loguru import logger
from typing import Any

class TelegramAlert:
    """Telegram integration for sending alerts and reports."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    async def send_message(self, message: str) -> bool:
        """Sends a text message to the configured Telegram chat."""
        if not self.bot_token or not self.chat_id:
            logger.warning("TelegramAlert: Missing bot_token or chat_id.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, json=payload, timeout=10.0)
                response.raise_for_status()
                logger.debug("Telegram alert sent successfully.")
                return True
        except Exception as e:
            logger.error("Failed to send Telegram alert: {}", e)
            return False
