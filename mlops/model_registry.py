"""MLOps Model Registry for maintaining historical iteration models."""

import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime, timezone

class ModelRegistry:
    """A local registry to index and retrieve HMM models by UUID/Version."""
    
    def __init__(self, registry_dir: str = "saved_models"):
        self.registry_path = Path(registry_dir)
        self.metadata_file = self.registry_path / "registry.json"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"active_version": None, "models": []}

    def _save_registry(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def register_model(self, version: str, performance_metrics: Dict[str, Any], file_path: str):
        """Saves a model to the catalog."""
        model_entry = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": performance_metrics,
            "path": file_path
        }
        self.metadata["models"].append(model_entry)
        self._save_registry()
        logger.info(f"Registered new model version: {version}")

    def set_active_model(self, version: str):
        """Marks a model version as the primary active artifact for the bot."""
        available = [m["version"] for m in self.metadata["models"]]
        if version in available:
            self.metadata["active_version"] = version
            self._save_registry()
            logger.info(f"Promoted model {version} to ACTIVE.")
        else:
            logger.error(f"Cannot set active model. Version {version} not found.")

    def get_active_model(self) -> dict | None:
        if not self.metadata["active_version"]:
            return None
        return next((m for m in self.metadata["models"] if m["version"] == self.metadata["active_version"]), None)
