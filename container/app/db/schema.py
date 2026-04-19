"""SQLite table definitions and initialization.

Manages the SQLite database schema for all structured data: configuration,
secrets, user accounts, conversation history, and analytics.
"""

import asyncio

import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.config import settings

_write_conn: aiosqlite.Connection | None = None
_write_lock = asyncio.Lock()


def _db_path() -> Path:
    """Resolve the SQLite database path and ensure the parent directory exists."""
    p = Path(settings.sqlite_db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


async def _get_or_create_write_connection() -> aiosqlite.Connection:
    """Get or create the shared write connection."""
    global _write_conn
    if _write_conn is None:
        _write_conn = await aiosqlite.connect(str(_db_path()))
        _write_conn.row_factory = aiosqlite.Row
        await _write_conn.execute("PRAGMA journal_mode=WAL")
        await _write_conn.execute("PRAGMA foreign_keys=ON")
    return _write_conn


@asynccontextmanager
async def get_db_read() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager returning a per-call read-only database connection.

    A fresh ``aiosqlite`` connection is opened for every read scope and
    closed on exit. WAL mode is persistent on the database file (set on
    the write connection at startup), so concurrent readers do not block
    each other and do not block writers. ``PRAGMA query_only=ON`` enforces
    read-only access at the connection level.
    """
    db = await aiosqlite.connect(str(_db_path()))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA query_only=ON")
    try:
        yield db
    finally:
        await db.close()


@asynccontextmanager
async def get_db_write() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager returning the write database connection.

    Acquires _write_lock to serialize writes.
    """
    async with _write_lock:
        db = await _get_or_create_write_connection()
        yield db


# Backward-compatible alias -- points to the write path (safe default).
get_db = get_db_write


async def close_db() -> None:
    """Close the shared write connection. Call on shutdown."""
    global _write_conn
    if _write_conn is not None:
        await _write_conn.close()
        _write_conn = None


async def init_db() -> None:
    """Initialize database schema and seed default data.

    Called at container startup. All operations are idempotent.
    """
    async with get_db() as db:
        await _create_tables(db)
        await _create_indexes(db)
        await _seed_defaults(db)
        await _run_migrations(db)
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
            temperature REAL NOT NULL DEFAULT 0.2,
            max_tokens INTEGER NOT NULL DEFAULT 1024,
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
        CREATE TABLE IF NOT EXISTS agent_mcp_tools (
            agent_id TEXT NOT NULL,
            server_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            PRIMARY KEY (agent_id, server_name, tool_name)
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
            end_time TEXT,
            duration_ms REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'ok',
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS trace_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL UNIQUE,
            conversation_id TEXT,
            user_input TEXT,
            final_response TEXT,
            agents TEXT,
            total_duration_ms REAL,
            label TEXT,
            source TEXT,
            routing_agent TEXT,
            routing_confidence REAL,
            routing_duration_ms REAL,
            routing_reasoning TEXT,
            agent_instructions TEXT,
            conversation_turns TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS send_device_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            device_type TEXT NOT NULL CHECK(device_type IN ('notify', 'tts')),
            ha_service_target TEXT NOT NULL,
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
        "CREATE INDEX IF NOT EXISTS idx_agent_mcp_tools_agent "
        "ON agent_mcp_tools(agent_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_spans_trace_id "
        "ON trace_spans(trace_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_spans_created_at "
        "ON trace_spans(created_at)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_summary_trace_id "
        "ON trace_summary(trace_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_summary_created_at "
        "ON trace_summary(created_at)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_summary_routing_agent "
        "ON trace_summary(routing_agent)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_trace_summary_label "
        "ON trace_summary(label)"
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
        ("cache.response.enabled", "true", "bool", "cache", "Enable response cache storage"),
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
        ("rewrite.model", "groq/llama-3.1-8b-instant", "string", "rewrite", "LLM model for rewrite agent"),
        ("rewrite.temperature", "0.8", "float", "rewrite", "Temperature for rewrite agent"),
        # Personality settings
        ("personality.prompt", "", "string", "personality", "Personality system prompt for response mediation"),
        # Communication settings
        ("communication.streaming_mode", "websocket", "string", "communication", "Streaming mode: websocket, sse, none"),
        ("communication.ws_reconnect_interval", "5", "int", "communication", "WebSocket reconnect interval in seconds"),
        ("communication.stream_buffer_size", "1", "int", "communication", "Token batching buffer size"),
        # A2A settings
        ("a2a.default_timeout", "10", "int", "a2a", "Default agent timeout in seconds"),
        ("a2a.max_iterations", "3", "int", "a2a", "Max iterations per agent to prevent loops"),
        # General settings
        ("general.conversation_context_turns", "3", "int", "general", "Number of conversation turns to keep"),
        # Home context settings
        ("home.timezone", "", "string", "home", "Manual timezone override (e.g., Europe/Berlin). Empty = auto-detect from HA."),
        ("home.location_name", "", "string", "home", "Manual home location name override. Empty = auto-detect from HA."),
    ]

    await db.executemany(
        "INSERT OR IGNORE INTO settings (key, value, value_type, category, description) "
        "VALUES (?, ?, ?, ?, ?)",
        default_settings,
    )

    # Default agent configs
    default_agents = [
        ("orchestrator", 1, "groq/llama-3.1-8b-instant", 10, 3, 0.3, 1024, "Intent classification and task routing"),
        ("light-agent", 1, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Lighting control"),
        ("music-agent", 1, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Music and media playback"),
        ("general-agent", 1, "openrouter/openai/gpt-4o-mini", 5, 3, 0.5, 1024, "Fallback and general Q&A"),
        ("timer-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Timers and alarms"),
        ("climate-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Climate and HVAC control"),
        ("media-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Media player control"),
        ("scene-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Scene activation"),
        ("automation-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Automation management"),
        ("security-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 3, 0.2, 1024, "Security system control"),
        ("send-agent", 0, "openrouter/openai/gpt-4o-mini", 5, 1, 0.2, 512, "Send content to devices via notification or TTS"),
        ("rewrite-agent", 0, "groq/llama-3.1-8b-instant", 2, 1, 0.8, 1024, "Cached response phrasing variation"),
        ("filler-agent", 1, "groq/llama-3.1-8b-instant", 3, 1, 0.7, 50, "Interim filler TTS phrase generation"),
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

    # Default entity visibility rules
    default_visibility_rules = [
        ("light-agent", "domain_include", "light"),
        ("light-agent", "domain_include", "switch"),
        ("music-agent", "domain_include", "media_player"),
        ("climate-agent", "domain_include", "climate"),
        ("climate-agent", "domain_include", "weather"),
        ("media-agent", "domain_include", "media_player"),
        ("scene-agent", "domain_include", "scene"),
        ("automation-agent", "domain_include", "automation"),
        ("timer-agent", "domain_include", "timer"),
        ("timer-agent", "domain_include", "input_datetime"),
        ("timer-agent", "domain_include", "persistent_notification"),
        ("timer-agent", "domain_include", "media_player"),
        ("timer-agent", "domain_include", "calendar"),
        ("security-agent", "domain_include", "alarm_control_panel"),
        ("security-agent", "domain_include", "lock"),
        # Sensor device_class rules for specialist agents
        ("climate-agent", "domain_include", "sensor"),
        ("climate-agent", "device_class_include", "temperature"),
        ("climate-agent", "device_class_include", "humidity"),
        ("climate-agent", "device_class_include", "pressure"),
        ("climate-agent", "device_class_include", "dew_point"),
        ("climate-agent", "device_class_include", "atmospheric_pressure"),
        ("climate-agent", "device_class_include", "moisture"),
        ("climate-agent", "device_class_include", "precipitation_intensity"),
        ("climate-agent", "device_class_include", "wind_speed"),
        ("climate-agent", "device_class_include", "wind_direction"),
        ("security-agent", "domain_include", "sensor"),
        ("security-agent", "domain_include", "binary_sensor"),
        ("security-agent", "device_class_include", "motion"),
        ("security-agent", "device_class_include", "occupancy"),
        ("security-agent", "device_class_include", "door"),
        ("security-agent", "device_class_include", "window"),
        ("security-agent", "device_class_include", "tamper"),
        ("security-agent", "device_class_include", "vibration"),
        ("security-agent", "device_class_include", "smoke"),
        ("security-agent", "device_class_include", "gas"),
        ("security-agent", "device_class_include", "carbon_monoxide"),
        ("security-agent", "device_class_include", "doorbell"),
        ("security-agent", "device_class_include", "opening"),
        ("security-agent", "device_class_include", "safety"),
        ("light-agent", "domain_include", "sensor"),
        ("light-agent", "device_class_include", "illuminance"),
    ]

    await db.executemany(
        "INSERT OR IGNORE INTO entity_visibility_rules (agent_id, rule_type, rule_value) "
        "VALUES (?, ?, ?)",
        default_visibility_rules,
    )


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """Run incremental schema migrations based on schema_version."""
    cursor = await db.execute("SELECT MAX(version) FROM schema_version")
    row = await cursor.fetchone()
    current_version = row[0] if row and row[0] else 1

    if current_version < 2:
        # Migration 2: Lower default temperature for action-oriented agents
        await db.execute("""
            UPDATE agent_configs
            SET temperature = 0.2
            WHERE agent_id IN (
                'light-agent', 'music-agent', 'timer-agent',
                'climate-agent', 'media-agent', 'scene-agent',
                'automation-agent', 'security-agent'
            )
            AND temperature = 0.7
        """)
        await db.execute("""
            UPDATE agent_configs
            SET temperature = 0.5
            WHERE agent_id = 'general-agent'
            AND temperature = 0.7
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (2)"
        )

    if current_version < 3:
        # Migration 3: Increase light-agent max_tokens and default timeout
        await db.execute("""
            UPDATE agent_configs
            SET max_tokens = 512
            WHERE agent_id = 'light-agent'
            AND max_tokens = 256
        """)
        await db.execute("""
            UPDATE settings
            SET value = '10'
            WHERE key = 'a2a.default_timeout'
            AND value = '5'
        """)
        await db.execute("""
            UPDATE agent_configs
            SET max_tokens = 512
            WHERE agent_id = 'music-agent'
            AND max_tokens = 256
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (3)"
        )

    if current_version < 4:
        # Migration 4: Seed default entity visibility rules for agents with zero rules
        agents_with_defaults = {
            "light-agent": ["light", "switch"],
            "music-agent": ["media_player"],
            "climate-agent": ["climate"],
            "media-agent": ["media_player"],
            "scene-agent": ["scene"],
            "automation-agent": ["automation"],
            "security-agent": ["alarm_control_panel", "lock"],
        }

        for agent_id, domains in agents_with_defaults.items():
            cursor = await db.execute(
                "SELECT COUNT(*) FROM entity_visibility_rules WHERE agent_id = ?",
                (agent_id,),
            )
            row = await cursor.fetchone()
            if row and row[0] == 0:
                for domain in domains:
                    await db.execute(
                        "INSERT OR IGNORE INTO entity_visibility_rules "
                        "(agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
                        (agent_id, "domain_include", domain),
                    )

        # Migrate legacy rule_types
        await db.execute(
            "UPDATE entity_visibility_rules SET rule_type = 'entity_include' "
            "WHERE rule_type = 'entity'"
        )
        await db.execute(
            "UPDATE entity_visibility_rules SET rule_type = 'domain_include' "
            "WHERE rule_type = 'domain'"
        )
        await db.execute(
            "UPDATE entity_visibility_rules SET rule_type = 'area_include' "
            "WHERE rule_type = 'area'"
        )

        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (4)"
        )

    if current_version < 5:
        # Migration 5: Increase rewrite-agent max_tokens from 128 to 512
        await db.execute("""
            UPDATE agent_configs
            SET max_tokens = 512
            WHERE agent_id = 'rewrite-agent'
            AND max_tokens IN (128, 256)
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (5)"
        )

    if current_version < 6:
        # Migration 6: Upgrade timer/scene/automation agents to ActionableAgent
        # Increase max_tokens from 256 to 512
        await db.execute("""
            UPDATE agent_configs
            SET max_tokens = 512
            WHERE agent_id IN ('timer-agent', 'scene-agent', 'automation-agent')
            AND max_tokens = 256
        """)
        # Add timer-agent visibility rules
        await db.execute(
            "INSERT OR IGNORE INTO entity_visibility_rules "
            "(agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
            ("timer-agent", "domain_include", "timer"),
        )
        await db.execute(
            "INSERT OR IGNORE INTO entity_visibility_rules "
            "(agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
            ("timer-agent", "domain_include", "input_datetime"),
        )
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (6)"
        )

    if current_version < 7:
        # Migration 7: Upgrade media-agent to ActionableAgent
        # Increase max_tokens from 256 to 512
        await db.execute("""
            UPDATE agent_configs
            SET max_tokens = 512
            WHERE agent_id = 'media-agent'
            AND max_tokens = 256
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (7)"
        )

    if current_version < 8:
        # Migration 8: Timer agent extensions -- add visibility for notification, media_player, calendar
        new_rules = [
            ("timer-agent", "domain_include", "persistent_notification"),
            ("timer-agent", "domain_include", "media_player"),
            ("timer-agent", "domain_include", "calendar"),
        ]
        await db.executemany(
            "INSERT OR IGNORE INTO entity_visibility_rules (agent_id, rule_type, rule_value) "
            "VALUES (?, ?, ?)",
            new_rules,
        )
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (8)"
        )

    if current_version < 9:
        # Migration 9: Add conversation_turns column to trace_summary
        try:
            await db.execute(
                "ALTER TABLE trace_summary ADD COLUMN conversation_turns TEXT"
            )
        except Exception:
            pass  # Column may already exist
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (9)"
        )

    if current_version < 10:
        # Migration 10: Increase max_tokens to prevent response truncation
        await db.execute("""
            UPDATE agent_configs SET max_tokens = 1024
            WHERE max_tokens IN (256, 512)
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (10)"
        )

    if current_version < 11:
        # Migration 11: Add reasoning_effort column to agent_configs
        try:
            await db.execute(
                "ALTER TABLE agent_configs ADD COLUMN reasoning_effort TEXT"
            )
        except Exception:
            pass  # Column may already exist
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (11)"
        )

    if current_version < 12:
        # Migration 12: Send device mappings for send-agent
        await db.execute("""
            CREATE TABLE IF NOT EXISTS send_device_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                device_type TEXT NOT NULL CHECK(device_type IN ('notify', 'tts')),
                ha_service_target TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (12)"
        )

    if current_version < 13:
        # Migration 13: Add end_time column to trace_spans
        try:
            await db.execute(
                "ALTER TABLE trace_spans ADD COLUMN end_time TEXT"
            )
        except Exception:
            pass  # Column may already exist
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (13)"
        )

    if current_version < 14:
        # Migration 14: Ensure device_class_include rules exist for all agents
        dc_rules = [
            # climate-agent sensor filtering
            ("climate-agent", "device_class_include", "temperature"),
            ("climate-agent", "device_class_include", "humidity"),
            ("climate-agent", "device_class_include", "pressure"),
            ("climate-agent", "device_class_include", "dew_point"),
            ("climate-agent", "device_class_include", "atmospheric_pressure"),
            ("climate-agent", "device_class_include", "moisture"),
            ("climate-agent", "device_class_include", "precipitation_intensity"),
            ("climate-agent", "device_class_include", "wind_speed"),
            ("climate-agent", "device_class_include", "wind_direction"),
            # security-agent sensor filtering
            ("security-agent", "device_class_include", "motion"),
            ("security-agent", "device_class_include", "occupancy"),
            ("security-agent", "device_class_include", "door"),
            ("security-agent", "device_class_include", "window"),
            ("security-agent", "device_class_include", "tamper"),
            ("security-agent", "device_class_include", "vibration"),
            ("security-agent", "device_class_include", "smoke"),
            ("security-agent", "device_class_include", "gas"),
            ("security-agent", "device_class_include", "carbon_monoxide"),
            ("security-agent", "device_class_include", "doorbell"),
            ("security-agent", "device_class_include", "opening"),
            ("security-agent", "device_class_include", "safety"),
            # light-agent sensor filtering
            ("light-agent", "device_class_include", "illuminance"),
        ]
        await db.executemany(
            "INSERT OR IGNORE INTO entity_visibility_rules "
            "(agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
            dc_rules,
        )
        # Also ensure domain_include sensor rules exist for agents that need device_class filtering
        sensor_domain_rules = [
            ("climate-agent", "domain_include", "sensor"),
            ("security-agent", "domain_include", "sensor"),
            ("security-agent", "domain_include", "binary_sensor"),
            ("light-agent", "domain_include", "sensor"),
        ]
        await db.executemany(
            "INSERT OR IGNORE INTO entity_visibility_rules "
            "(agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
            sensor_domain_rules,
        )
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (14)"
        )

    if current_version < 15:
        # Migration 15: Add weather domain visibility for climate-agent
        await db.execute(
            "INSERT OR IGNORE INTO entity_visibility_rules "
            "(agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
            ("climate-agent", "domain_include", "weather"),
        )
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (15)"
        )
