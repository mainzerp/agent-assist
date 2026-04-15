"""Database access layer for agent-assist.

Provides async SQLite operations for all structured data.
"""

from app.db.schema import get_db, get_db_read, get_db_write, init_db
from app.db.repository import (
    AdminAccountRepository,
    AgentConfigRepository,
    AliasRepository,
    AnalyticsRepository,
    ConversationRepository,
    CustomAgentRepository,
    EntityMatchingConfigRepository,
    EntityVisibilityRepository,
    McpServerRepository,
    PluginRepository,
    SecretsRepository,
    SetupStateRepository,
    SettingsRepository,
)

__all__ = [
    "get_db",
    "init_db",
    "AdminAccountRepository",
    "AgentConfigRepository",
    "AliasRepository",
    "AnalyticsRepository",
    "ConversationRepository",
    "CustomAgentRepository",
    "EntityMatchingConfigRepository",
    "EntityVisibilityRepository",
    "McpServerRepository",
    "PluginRepository",
    "SecretsRepository",
    "SetupStateRepository",
    "SettingsRepository",
]
