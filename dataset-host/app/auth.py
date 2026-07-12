import secrets

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_read_access(x_api_key: str | None = Header(default=None)) -> None:
    configured = get_settings().read_api_key
    if not configured:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
