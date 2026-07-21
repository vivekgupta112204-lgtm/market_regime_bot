"""Internal process and network integrity watchdog."""

import asyncio
from loguru import logger
import httpx
from maintenance.recovery import RecoveryManager

class WatchdogService:
    """Independent service polling for health failures and initiating recovery."""
    
    def __init__(self, api_port: int = 8000):
        self.api_url = f"http://localhost:{api_port}/health"
        self.recovery = RecoveryManager()
        self._running = False
        
    async def start_polling(self, interval_seconds: int = 60):
        self._running = True
        logger.info("Watchdog started.")
        
        async with httpx.AsyncClient() as client:
            while self._running:
                await asyncio.sleep(interval_seconds)
                try:
                    resp = await client.get(self.api_url, timeout=5.0)
                    if resp.status_code != 200:
                        logger.error(f"Watchdog detected API unhealthy ({resp.status_code}).")
                        await self.recovery.handle_api_failure()
                except Exception as e:
                    logger.critical(f"Watchdog catastrophic network failure hitting local API: {e}")
                    await self.recovery.handle_network_failure()
