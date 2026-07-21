"""Hooks for triggering the auto-retrainer from MLOps bounds."""

import asyncio
from loguru import logger
from automation.retrainer import AutoRetrainer

async def hook_auto_retrainer(trigger: str):
    """Bridge function that integrates MLOps with Automation pipelines."""
    retrainer = AutoRetrainer()
    await retrainer.trigger_retraining(trigger_reason=trigger)
