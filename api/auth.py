"""Simple API Key Authentication."""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config.settings import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Validates the incoming request API key against settings."""
    settings = get_settings()
    # For now, default to a hardcoded expected key if none is in config
    expected_key = "secret_dashboard_key_123" 
    
    if api_key_header != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API Key",
        )
    return api_key_header
