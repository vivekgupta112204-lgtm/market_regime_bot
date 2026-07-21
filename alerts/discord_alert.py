from __future__ import annotations

import httpx
from loguru import logger
from typing import Any

class DiscordAlert:
    """Discord Webhook integration for sending alerts and embedded reports."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_message(self, message: str, title: str | None = None, color: int = 0x00FF00) -> bool:
        """Sends a message or rich embed to a Discord channel."""
        if not self.webhook_url:
            logger.warning("DiscordAlert: Missing webhook URL.")
            return False

        embed = {
            "description": message,
            "color": color
        }
        if title:
            embed["title"] = title

        payload = {
            "embeds": [embed]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload, timeout=10.0)
                response.raise_for_status()
                logger.debug("Discord alert sent successfully.")
                return True
        except Exception as e:
            logger.error("Failed to send Discord alert: {}", e)
            return False
