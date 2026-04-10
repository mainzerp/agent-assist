import os
import logging
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.db.repository import SecretsRepository

logger = logging.getLogger(__name__)

FERNET_KEY_PATH = Path("/data/.fernet_key")

_fernet: Fernet | None = None


def _load_or_generate_key() -> bytes:
    if FERNET_KEY_PATH.exists():
        key = FERNET_KEY_PATH.read_bytes().strip()
        logger.info("Fernet key loaded from file")
        return key
    key = Fernet.generate_key()
    FERNET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FERNET_KEY_PATH.write_bytes(key)
    FERNET_KEY_PATH.chmod(0o600)
    logger.info("New Fernet key generated")
    return key


def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_load_or_generate_key())
    return _fernet


def get_fernet_key() -> bytes:
    """Return the raw Fernet key bytes (for deriving secondary keys)."""
    return _load_or_generate_key()


def encrypt(plaintext: str) -> bytes:
    return get_fernet().encrypt(plaintext.encode("utf-8"))


def decrypt(ciphertext: bytes) -> str:
    try:
        return get_fernet().decrypt(ciphertext).decode("utf-8")
    except InvalidToken:
        logger.warning("Failed to decrypt secret -- key may have changed")
        raise ValueError("Decryption failed")


async def store_secret(key: str, plaintext: str) -> None:
    encrypted = encrypt(plaintext)
    await SecretsRepository.set(key, encrypted)


async def retrieve_secret(key: str) -> str | None:
    encrypted = await SecretsRepository.get(key)
    if encrypted is None:
        return None
    return decrypt(encrypted)


async def delete_secret(key: str) -> None:
    await SecretsRepository.delete(key)
