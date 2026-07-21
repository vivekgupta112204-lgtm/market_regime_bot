"""Automated updates processing (pulling latest container or git codebase)."""

import subprocess
import threading
from loguru import logger

class SystemUpdater:
    """Safely updates the codebase. Usually triggered in a safe window."""
    
    def check_for_updates(self) -> bool:
        """Fetches the latest git objects and checks if HEAD is behind origin."""
        try:
            subprocess.run(["git", "fetch"], check=True, capture_output=True)
            result = subprocess.run(["git", "status", "-uno"], check=True, text=True, capture_output=True)
            if "Your branch is behind" in result.stdout:
                return True
            return False
        except Exception:
            return False

    def pull_and_restart(self):
        """Pulls latest code and halts process so Systemd/Docker can restart it."""
        logger.warning("System updater triggered. Pulling latest code.")
        try:
            import os
            subprocess.run(["git", "pull", "--rebase"], check=True)
            logger.warning("Code updated. Issuing restart via system exit.")
            os._exit(0) # Let the container Orchestrator (Docker/K8s) revive us
        except Exception as e:
            logger.error(f"Auto-update failed: {e}")

    def run_update_async_if_available(self):
        if self.check_for_updates():
            threading.Thread(target=self.pull_and_restart, daemon=True).start()
