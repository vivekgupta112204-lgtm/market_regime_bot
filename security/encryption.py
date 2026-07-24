"""Encryption routines for local secure storage."""

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger

def _generate_key_from_password(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

class SystemEncryptor:
    """Safely encrypts API keys and secrets at rest using AES (via Fernet)."""
    
    def __init__(self, master_password: str | None = None):
        salt_file = "security/.vault_salt"
        if os.path.exists(salt_file):
            with open(salt_file, "rb") as f:
                self.salt = f.read()
        else:
            self.salt = os.urandom(16)
            os.makedirs(os.path.dirname(salt_file), exist_ok=True)
            with open(salt_file, "wb") as f:
                f.write(self.salt)
        
        pwd = master_password or os.getenv("HMM_MASTER_PW")
        if not pwd:
            raise RuntimeError("FATAL: HMM_MASTER_PW environment variable not set. Refusing to initialize encryptor.")
            
        key = _generate_key_from_password(pwd, self.salt)
        self.fernet = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Returns encrypted string safe for file storage."""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Restores encrypted string."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
