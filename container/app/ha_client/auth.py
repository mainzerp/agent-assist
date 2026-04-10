"""HA authentication and token management."""

import logging

from app.security.encryption import retrieve_secret, store_secret

logger = logging.getLogger(__name__)

HA_TOKEN_SECRET_KEY = "ha_token"


async def get_ha_token() -> str | None:
    return await retrieve_secret(HA_TOKEN_SECRET_KEY)


async def set_ha_token(token: str) -> None:
    await store_secret(HA_TOKEN_SECRET_KEY, token)
    logger.info("HA token updated")


def build_auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def get_auth_headers() -> dict[str, str] | None:
    token = await get_ha_token()
    if token is None:
        return None
    return build_auth_headers(token)
