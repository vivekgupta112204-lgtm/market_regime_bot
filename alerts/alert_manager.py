from __future__ import annotations

import asyncio
from loguru import logger
from typing import Any
from pathlib import Path

from config.settings import get_settings
from alerts.telegram_alert import TelegramAlert
from alerts.discord_alert import DiscordAlert
from alerts.email_alert import EmailAlert

class AlertManager:
    """Centralized alerting system managing all notification channels."""

    def __init__(self):
        settings = get_settings()
        keys = settings.api_keys
        
        self.telegram = TelegramAlert(
            bot_token=keys.dict().get("telegram_bot_token", ""),
            chat_id=keys.dict().get("telegram_chat_id", "")
        ) if keys.dict().get("telegram_bot_token") else None

        self.discord = DiscordAlert(
            webhook_url=keys.dict().get("discord_webhook_url", "")
        ) if keys.dict().get("discord_webhook_url") else None
        
        # Email settings are hypothetical based on common keys
        self.email = None

    async def broadcast(self, message: str, title: str | None = None, level: str = "INFO"):
        """Sends an alert to all configured async channels."""
        logger.info(f"Broadcasting Alert [{level}]: {message}")
        
        tasks = []
        if self.telegram:
            tasks.append(self.telegram.send_message(f"<b>{title}</b>\n{message}" if title else message))
        if self.discord:
            color = 0xFF0000 if level == "ERROR" else (0xFFA500 if level == "WARNING" else 0x00FF00)
            tasks.append(self.discord.send_message(message, title, color))
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def trigger_trade_executed(self, trade_info: dict[str, Any]):
        """Helper to format and schedule an execution alert."""
        msg = f"Order filled: {trade_info.get('action')} {trade_info.get('qty')} @ {trade_info.get('price')}"
        asyncio.create_task(self.broadcast(msg, title="Trade Executed", level="INFO"))

    def trigger_regime_change(self, old_regime: str, new_regime: str):
        msg = f"Market changed from {old_regime} to {new_regime}."
        asyncio.create_task(self.broadcast(msg, title="Regime Change Detected", level="WARNING"))

    def trigger_system_error(self, error_msg: str):
        asyncio.create_task(self.broadcast(error_msg, title="System Error", level="ERROR"))
