"""Pydantic settings configuration loader.

Loads infrastructure environment variables with sensible defaults.
All other configuration is stored in SQLite and managed via the admin UI.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Infrastructure settings loaded from environment variables.

    Only settings required to start the container process use env vars.
    All other settings (HA connection, LLM keys, agent config, etc.) are
    stored in SQLite and configured via the setup wizard or admin UI.
    """

    container_host: str = "0.0.0.0"
    container_port: int = 8080
    log_level: str = "INFO"
    chromadb_persist_dir: str = "/data/chromadb"
    sqlite_db_path: str = "/data/agent_assist.db"

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
