"""SQLite table definitions and initialization.

Manages the SQLite database schema for all structured data: configuration,
secrets, user accounts, conversation history, and analytics.
"""

import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.config import settings


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager for database connections."""
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    """Initialize database schema and seed default data.

    Called at container startup. All operations are idempotent.
    """
    async with get_db() as db:
        await _create_tables(db)
        await _create_indexes(db)
        await _seed_defaults(db)
        await db.commit()


async def _create_tables(db: aiosqlite.Connection) -> None:
    """Create all tables if they do not exist."""

    await db.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            value_type TEXT NOT NULL DEFAULT 'string',
            category TEXT NOT NULL DEFAULT 'general',
            description TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS agent_configs (
            agent_id TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL DEFAULT 1,
            model TEXT,
            timeout INTEGER NOT NULL DEFAULT 5,
            max_iterations INTEGER NOT NULL DEFAULT 3,
            temperature REAL NOT NULL DEFAULT 0.7,
            max_tokens INTEGER NOT NULL DEFAULT 256,
            description TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS custom_agents (
            name TEXT PRIMARY KEY,
            description TEXT,
            system_prompt TEXT NOT NULL,
            model_override TEXT,
            mcp_tools TEXT,
            entity_visibility TEXT,
            intent_patterns TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS entity_matching_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            description TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS mcp_servers (
            name TEXT PRIMARY KEY,
            transport TEXT NOT NULL,
            command_or_url TEXT NOT NULL,
            env_vars TEXT,
            timeout INTEGER NOT NULL DEFAULT 30,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS secrets (
            key TEXT PRIMARY KEY,
            encrypted_value BLOB NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS admin_accounts (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_login TEXT
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS setup_state (
            step TEXT PRIMARY KEY,
            completed INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS entity_visibility_rules (
            agent_id TEXT NOT NULL,
            rule_type TEXT NOT NULL,
            rule_value TEXT NOT NULL,
            PRIMARY KEY (agent_id, rule_type, rule_value)
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS plugins (
            name TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            version TEXT,
            description TEXT,
            loaded_at TEXT
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            user_text TEXT NOT NULL,
            agent_id TEXT,
            response_text TEXT,
            action_executed TEXT,
            cache_hit TEXT,
            latency_ms REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            agent_id TEXT,
            data TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS trace_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            span_name TEXT NOT NULL,
            agent_id TEXT,
            parent_span TEXT,
            start_time TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'ok',
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


async def _create_indexes(db: aiosqlite.Connection) -> None:
    """Create indexes for query performance."""
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_aliases_entity_id ON aliases(entity_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id "
        "ON conversations(conversation_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_created_at "
        "ON conversations(created_at)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_agent_id "
        "ON conversations(agent_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_event_type "
        "ON analytics(event_type)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_analytics_created_at "
        "ON analytics(created_at)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_entity_visibility_agent "
        "ON entity_visibility_rules(agent_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_servers_enabled "
        "ON mcp_servers(enabled)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_spans_trace_id "
        "ON trace_spans(trace_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_spans_created_at "
        "ON trace_spans(created_at)"
    )


async def _seed_defaults(db: aiosqlite.Connection) -> None:
    """Insert default seed data. Uses INSERT OR IGNORE to be idempotent."""

    # Default settings
    default_settings = [
        # Cache settings
        ("cache.routing.threshold", "0.92", "float", "cache", "Routing cache hit threshold"),
        ("cache.routing.max_entries", "50000", "int", "cache", "Routing cache max entries (LRU eviction)"),
        ("cache.response.threshold", "0.95", "float", "cache", "Response cache hit threshold"),
        ("cache.response.partial_threshold", "0.80", "float", "cache", "Response cache partial match threshold"),
        ("cache.response.max_entries", "20000", "int", "cache", "Response cache max entries (LRU eviction)"),
        # Embedding settings
        ("embedding.provider", "local", "string", "embedding", "Embedding provider: local, openrouter, groq, anthropic, or ollama"),
        ("embedding.local_model", "all-MiniLM-L6-v2", "string", "embedding", "Local embedding model name"),
        ("embedding.external_model", "", "string", "embedding", "External embedding model (e.g., openai/text-embedding-3-small)"),
        ("embedding.dimension", "384", "int", "embedding", "Embedding dimension (auto-detected from model)"),
        # Entity matching settings
        ("entity_matching.confidence_threshold", "0.75", "float", "entity_matching", "Minimum confidence for entity match"),
        ("entity_matching.top_n_candidates", "3", "int", "entity_matching", "Top-N candidates for LLM disambiguation"),
        # Presence settings
        ("presence.enabled", "true", "bool", "presence", "Enable presence detection"),
        ("presence.decay_timeout", "300", "int", "presence", "Presence decay timeout in seconds"),
        # Rewrite agent settings
        ("rewrite.enabled", "false", "bool", "rewrite", "Enable rewrite agent for cached response variation"),
        ("rewrite.model", "groq/llama-3.1-8b-instant", "string", "rewrite", "LLM model for rewrite agent"),
        ("rewrite.temperature", "0.8", "float", "rewrite", "Temperature for rewrite agent"),
        # Communication settings
        ("communication.streaming_mode", "websocket", "string", "communication", "Streaming mode: websocket, sse, none"),
        ("communication.ws_reconnect_interval", "5", "int", "communication", "WebSocket reconnect interval in seconds"),
        ("communication.stream_buffer_size", "1", "int", "communication", "Token batching buffer size"),
        # A2A settings
        ("a2a.default_timeout", "5", "int", "a2a", "Default agent timeout in seconds"),
        ("a2a.max_iterations", "3", "int", "a2a", "Max iterations per agent to prevent loops"),
        # General settings
        ("general.conversation_context_turns", "3", "int", "general", "Number of conversation turns to keep"),
    ]

    await db.executemany(
        "INSERT OR IGNORE INTO settings (key, value, value_type, category, description) "
        "VALUES (?, ?, ?, ?, ?)",
        default_settings,
    )

    # Default agent configs
    default_agents = [
        ("orchestrator", 1, "groq/llama-3.1-8b-instant", 10, 3, 0.3, 256, "Intent classification and task routing"),
        ("light-agent", 1, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Lighting control"),
        ("music-agent", 1, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Music and media playback"),
        ("general-agent", 1, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 512, "Fallback and general Q&A"),
        ("timer-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Timers and alarms"),
        ("climate-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Climate and HVAC control"),
        ("media-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Media player control"),
        ("scene-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Scene activation"),
        ("automation-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Automation management"),
        ("security-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.7, 256, "Security system control"),
        ("rewrite-agent", 0, "groq/llama-3.1-8b-instant", 2, 1, 0.8, 128, "Cached response phrasing variation"),
    ]

    await db.executemany(
        "INSERT OR IGNORE INTO agent_configs "
        "(agent_id, enabled, model, timeout, max_iterations, temperature, max_tokens, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        default_agents,
    )

    # Default entity matching weights
    default_matching = [
        ("weight.levenshtein", "0.20", "Levenshtein distance signal weight"),
        ("weight.jaro_winkler", "0.20", "Jaro-Winkler similarity signal weight"),
        ("weight.phonetic", "0.15", "Phonetic matching signal weight"),
        ("weight.embedding", "0.30", "Embedding similarity signal weight"),
        ("weight.alias", "0.15", "Alias resolution signal weight"),
    ]

    await db.executemany(
        "INSERT OR IGNORE INTO entity_matching_config (key, value, description) "
        "VALUES (?, ?, ?)",
        default_matching,
    )

    # Setup wizard steps
    setup_steps = [
        ("admin_password",),
        ("ha_connection",),
        ("container_api_key",),
        ("llm_providers",),
        ("review_complete",),
    ]

    await db.executemany(
        "INSERT OR IGNORE INTO setup_state (step) VALUES (?)",
        setup_steps,
    )

    # Initial schema version
    await db.execute(
        "INSERT OR IGNORE INTO schema_version (version) VALUES (1)"
    )
