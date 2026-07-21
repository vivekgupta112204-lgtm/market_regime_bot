"""FastAPI Security Middlewares (JWT/Rate Limits)."""

from fastapi import Request, HTTPException, status
import time

# Simple mock in-memory rate limiter
RATE_LIMITS = {}
MAX_REQUESTS_PER_MINUTE = 60

async def rate_limit_middleware(request: Request, call_next):
    """Provides basic DDOS protection on API endpoints."""
    ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if ip not in RATE_LIMITS:
        RATE_LIMITS[ip] = []
    
    # Filter out timestamps older than 60 seconds
    RATE_LIMITS[ip] = [ts for ts in RATE_LIMITS[ip] if current_time - ts < 60.0]
    
    if len(RATE_LIMITS[ip]) >= MAX_REQUESTS_PER_MINUTE:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded."
        )
        
    RATE_LIMITS[ip].append(current_time)
    response = await call_next(request)
    return response

# Also JWT mocking block would go here, usually integrating `python-jose`
