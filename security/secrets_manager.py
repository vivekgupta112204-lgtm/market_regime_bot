"""Secrets Manager handling API Keys securely."""

import json
from pathlib import Path
from loguru import logger
from security.encryption import SystemEncryptor

class SecretsManager:
    """Manages secure reading/writing of Broker and API keys."""

    def __init__(self, vault_path: str = "config/.secrets.enc", master_password: str | None = None):
        self.vault_path = Path(vault_path)
        self.vault_path.parent.mkdir(exist_ok=True, parents=True)
        self.encryptor = SystemEncryptor(master_password)

    def store_secrets(self, secrets_dict: dict):
        """Encrypts and dumps a JSON dictionary containing secrets."""
        plain_text = json.dumps(secrets_dict)
        encrypted_text = self.encryptor.encrypt(plain_text)
        
        with open(self.vault_path, "w") as f:
            f.write(encrypted_text)
        logger.info("Secrets successfully vaulted.")

    def load_secrets(self) -> dict:
        """Loads and decrypts secrets if the vault exists."""
        if not self.vault_path.exists():
            return {}
            
        try:
            with open(self.vault_path, "r") as f:
                encrypted_text = f.read()
            plain_text = self.encryptor.decrypt(encrypted_text)
            return json.loads(plain_text)
        except Exception as e:
            logger.error(f"Failed to unlock secrets vault: {e}")
            return {}
