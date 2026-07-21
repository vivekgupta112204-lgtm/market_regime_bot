"""Database and artifact backup routines."""

import shutil
import zipfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

class BackupManager:
    """Handles compressed backups of critical bot state data."""
    
    def __init__(self, output_dir: str = "backups"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_dirs = ["data", "saved_models", "logs", "reports"]

    def perform_backup(self) -> str | None:
        """Compresses all local persistent state directories into a timestamped ZIP."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = self.output_dir / f"bot_backup_{timestamp}.zip"
        
        logger.info(f"Starting system backup: {backup_name}")
        try:
            with zipfile.ZipFile(backup_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for target in self.target_dirs:
                    target_path = Path(target)
                    if not target_path.exists():
                        continue
                    for root, dirs, files in target_path.walk():
                        for file in files:
                            # Avoid backing up pycache
                            if "__pycache__" not in root:
                                f_path = root / file
                                zipf.write(f_path, arcname=f_path.relative_to(target_path.parent))
            logger.info("System backup completed successfully.")
            return str(backup_name)
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None

    def backup_async(self):
        """Dispatches backup to background thread."""
        threading.Thread(target=self.perform_backup, daemon=True).start()
