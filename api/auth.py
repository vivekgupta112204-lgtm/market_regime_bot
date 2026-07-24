"""Simple API Key Authentication."""

import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config.settings import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _get_api_key():
    key = os.environ.get("DASHBOARD_API_KEY")
    # Fail loudly if key is absent.
    if not key:
        raise RuntimeError("FATAL: DASHBOARD_API_KEY environment variable is not set. Refusing to boot without secure auth.")
    return key

# Cache the expected key on startup
EXPECTED_API_KEY = _get_api_key()

async def verify_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Validates the incoming request API key against environmental secrets."""
    if not api_key_header or api_key_header != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API Key",
        )
    return api_key_header
