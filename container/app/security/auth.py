import hmac
import logging
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.responses import RedirectResponse
from starlette.requests import HTTPConnection
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from app.security.encryption import retrieve_secret, get_fernet_key
from app.security.hashing import verify_password
from app.db.repository import AdminAccountRepository

logger = logging.getLogger(__name__)

API_KEY_HEADER = "Authorization"
API_KEY_SECRET_NAME = "container_api_key"
SESSION_COOKIE_NAME = "agent_assist_session"
SESSION_MAX_AGE = 86400

_session_serializer: URLSafeTimedSerializer | None = None


def _get_session_serializer() -> URLSafeTimedSerializer:
    global _session_serializer
    if _session_serializer is None:
        import hashlib
        signing_key = hashlib.sha256(get_fernet_key()).hexdigest()
        _session_serializer = URLSafeTimedSerializer(signing_key)
    return _session_serializer


async def require_api_key(request: Request) -> str:
    auth_header = request.headers.get(API_KEY_HEADER)
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    provided_key = auth_header[7:]
    stored_key = await retrieve_secret(API_KEY_SECRET_NAME)
    if stored_key is None or not hmac.compare_digest(provided_key, stored_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return provided_key


async def require_api_key_ws(websocket: WebSocket) -> str:
    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get(API_KEY_HEADER)
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        await websocket.close(code=4001, reason="Unauthorized")
        raise HTTPException(status_code=401, detail="Unauthorized")
    stored_key = await retrieve_secret(API_KEY_SECRET_NAME)
    if stored_key is None or not hmac.compare_digest(token, stored_key):
        await websocket.close(code=4001, reason="Unauthorized")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token


async def require_admin_session(request: Request) -> dict:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie:
        raise HTTPException(status_code=401, detail="Session expired")
    try:
        data = _get_session_serializer().loads(cookie, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=401, detail="Session expired")
    return data


async def require_admin_session_redirect(request: Request) -> dict:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    is_htmx = request.headers.get("HX-Request") == "true"
    if not cookie:
        if is_htmx:
            return _htmx_redirect_response()
        raise HTTPException(
            status_code=303,
            headers={"Location": "/dashboard/login"},
            detail="Session expired",
        )
    try:
        data = _get_session_serializer().loads(cookie, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        if is_htmx:
            return _htmx_redirect_response()
        raise HTTPException(
            status_code=303,
            headers={"Location": "/dashboard/login"},
            detail="Session expired",
        )
    return data


def _htmx_redirect_response():
    """Return a 401 with HX-Redirect header so HTMX does a full page redirect."""
    raise HTTPException(
        status_code=401,
        headers={"HX-Redirect": "/dashboard/login"},
        detail="Session expired",
    )


async def authenticate_admin(username: str, password: str) -> dict | None:
    account = await AdminAccountRepository.get(username)
    if account is None:
        return None
    if not verify_password(password, account["password_hash"]):
        return None
    await AdminAccountRepository.update_last_login(username)
    return {"username": username}


def create_session_cookie(session_data: dict) -> str:
    return _get_session_serializer().dumps(session_data)
