"""Async CRUD operations via aiosqlite.

Provides repository classes for each SQLite table with typed
async methods for common operations.
"""

from __future__ import annotations

import contextlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from app.db.schema import get_db_read, get_db_write


def _now() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class SettingsRepository:
    """CRUD for the settings key-value store."""

    @staticmethod
    async def get(key: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT key, value, value_type, category, description FROM settings WHERE key = ?",
                (key,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    @staticmethod
    async def get_value(key: str, default: str | None = None) -> str | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = await cursor.fetchone()
            return row[0] if row else default

    @staticmethod
    async def set(
        key: str, value: str, value_type: str = "string", category: str = "general", description: str | None = None
    ) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO settings (key, value, value_type, category, description, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                (key, value, value_type, category, description, _now(), value, _now()),
            )
            await db.commit()

    @staticmethod
    async def get_by_category(category: str) -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT key, value, value_type, description FROM settings WHERE category = ?",
                (category,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def get_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT key, value, value_type, category, description FROM settings")
            return [dict(row) for row in await cursor.fetchall()]


class AgentConfigRepository:
    """CRUD for agent configurations."""

    @staticmethod
    async def get(agent_id: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM agent_configs WHERE agent_id = ?", (agent_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM agent_configs")
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def list_enabled() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM agent_configs WHERE enabled = 1")
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def upsert(agent_id: str, **kwargs: Any) -> None:
        allowed = {
            "enabled",
            "model",
            "timeout",
            "max_iterations",
            "temperature",
            "max_tokens",
            "description",
            "reasoning_effort",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = _now()

        columns = ", ".join(["agent_id", *list(fields.keys())])
        placeholders = ", ".join(["?"] * (len(fields) + 1))
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields)

        values = [agent_id, *list(fields.values())]
        async with get_db_write() as db:
            await db.execute(
                f"INSERT INTO agent_configs ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(agent_id) DO UPDATE SET {updates}",
                values,
            )
            await db.commit()


class SecretsRepository:
    """CRUD for Fernet-encrypted secrets."""

    @staticmethod
    async def get(key: str) -> bytes | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT encrypted_value FROM secrets WHERE key = ?", (key,))
            row = await cursor.fetchone()
            return row[0] if row else None

    @staticmethod
    async def set(key: str, encrypted_value: bytes) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO secrets (key, encrypted_value, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET encrypted_value=?, updated_at=?",
                (key, encrypted_value, _now(), encrypted_value, _now()),
            )
            await db.commit()

    @staticmethod
    async def delete(key: str) -> None:
        async with get_db_write() as db:
            await db.execute("DELETE FROM secrets WHERE key = ?", (key,))
            await db.commit()

    @staticmethod
    async def list_keys() -> list[str]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT key FROM secrets")
            return [row[0] for row in await cursor.fetchall()]


class AdminAccountRepository:
    """CRUD for admin accounts."""

    @staticmethod
    async def create(
        username: str,
        password_hash: str,
        *,
        force_overwrite: bool = False,
    ) -> None:
        """Create an admin account.

        ``force_overwrite=True`` uses ``INSERT OR REPLACE`` (only the
        one-time setup bootstrap should pass this). The default uses
        ``INSERT OR IGNORE`` so an authenticated session cannot silently
        overwrite an existing admin row via an unrelated code path.
        """
        verb = "INSERT OR REPLACE" if force_overwrite else "INSERT OR IGNORE"
        async with get_db_write() as db:
            await db.execute(
                f"{verb} INTO admin_accounts (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, _now()),
            )
            await db.commit()

    @staticmethod
    async def update_password(username: str, password_hash: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "UPDATE admin_accounts SET password_hash = ? WHERE username = ?",
                (password_hash, username),
            )
            await db.commit()

    @staticmethod
    async def get(username: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM admin_accounts WHERE username = ?", (username,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def update_last_login(username: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "UPDATE admin_accounts SET last_login = ? WHERE username = ?",
                (_now(), username),
            )
            await db.commit()

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT username, created_at, last_login FROM admin_accounts")
            return [dict(row) for row in await cursor.fetchall()]


class SetupStateRepository:
    """CRUD for setup wizard state tracking."""

    @staticmethod
    async def get_step(step: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM setup_state WHERE step = ?", (step,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def set_step_completed(step: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "UPDATE setup_state SET completed = 1, completed_at = ? WHERE step = ?",
                (_now(), step),
            )
            await db.commit()

    @staticmethod
    async def is_complete() -> bool:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM setup_state WHERE completed = 0")
            row = await cursor.fetchone()
            return row[0] == 0

    @staticmethod
    async def get_all_steps() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM setup_state")
            return [dict(row) for row in await cursor.fetchall()]


class AliasRepository:
    """CRUD for entity aliases."""

    @staticmethod
    async def get(alias: str) -> str | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT entity_id FROM aliases WHERE alias = ?", (alias,))
            row = await cursor.fetchone()
            return row[0] if row else None

    @staticmethod
    async def set(alias: str, entity_id: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO aliases (alias, entity_id, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(alias) DO UPDATE SET entity_id=?",
                (alias, entity_id, _now(), entity_id),
            )
            await db.commit()

    @staticmethod
    async def delete(alias: str) -> None:
        async with get_db_write() as db:
            await db.execute("DELETE FROM aliases WHERE alias = ?", (alias,))
            await db.commit()

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT alias, entity_id FROM aliases")
            return [dict(row) for row in await cursor.fetchall()]


class CustomAgentRepository:
    """CRUD for runtime-created custom agents."""

    @staticmethod
    async def get(name: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM custom_agents WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row is None:
                return None
            result = dict(row)
            for field in ("mcp_tools", "entity_visibility", "intent_patterns"):
                if result.get(field):
                    result[field] = json.loads(result[field])
            return result

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM custom_agents")
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                for field in ("mcp_tools", "entity_visibility", "intent_patterns"):
                    if row.get(field):
                        row[field] = json.loads(row[field])
            return rows

    @staticmethod
    async def list_enabled() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM custom_agents WHERE enabled = 1")
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                for field in ("mcp_tools", "entity_visibility", "intent_patterns"):
                    if row.get(field):
                        row[field] = json.loads(row[field])
            return rows

    @staticmethod
    async def create(name: str, system_prompt: str, **kwargs: Any) -> None:
        fields = {"description", "model_override", "mcp_tools", "entity_visibility", "intent_patterns", "enabled"}
        data = {k: v for k, v in kwargs.items() if k in fields}
        for field in ("mcp_tools", "entity_visibility", "intent_patterns"):
            if field in data and isinstance(data[field], (list, dict)):
                data[field] = json.dumps(data[field])

        columns = ", ".join(["name", "system_prompt", *list(data.keys())])
        placeholders = ", ".join(["?"] * (len(data) + 2))
        values = [name, system_prompt, *list(data.values())]

        async with get_db_write() as db:
            await db.execute(
                f"INSERT INTO custom_agents ({columns}) VALUES ({placeholders})",
                values,
            )
            await db.commit()

    @staticmethod
    async def update(name: str, **kwargs: Any) -> None:
        fields = {
            "description",
            "system_prompt",
            "model_override",
            "mcp_tools",
            "entity_visibility",
            "intent_patterns",
            "enabled",
        }
        data = {k: v for k, v in kwargs.items() if k in fields}
        if not data:
            return
        for field in ("mcp_tools", "entity_visibility", "intent_patterns"):
            if field in data and isinstance(data[field], (list, dict)):
                data[field] = json.dumps(data[field])
        data["updated_at"] = _now()

        set_clause = ", ".join(f"{k} = ?" for k in data)
        values = [*list(data.values()), name]

        async with get_db_write() as db:
            await db.execute(
                f"UPDATE custom_agents SET {set_clause} WHERE name = ?",
                values,
            )
            await db.commit()

    @staticmethod
    async def delete(name: str) -> None:
        async with get_db_write() as db:
            await db.execute("DELETE FROM custom_agents WHERE name = ?", (name,))
            await db.commit()


class McpServerRepository:
    """CRUD for MCP server configurations."""

    @staticmethod
    async def get(name: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM mcp_servers WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row is None:
                return None
            result = dict(row)
            if result.get("env_vars"):
                result["env_vars"] = json.loads(result["env_vars"])
            return result

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM mcp_servers")
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                if row.get("env_vars"):
                    row["env_vars"] = json.loads(row["env_vars"])
            return rows

    @staticmethod
    async def create(
        name: str, transport: str, command_or_url: str, env_vars: dict | None = None, timeout: int = 30
    ) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO mcp_servers (name, transport, command_or_url, env_vars, timeout) VALUES (?, ?, ?, ?, ?)",
                (name, transport, command_or_url, json.dumps(env_vars) if env_vars else None, timeout),
            )
            await db.commit()

    @staticmethod
    async def delete(name: str) -> None:
        async with get_db_write() as db:
            await db.execute("DELETE FROM mcp_servers WHERE name = ?", (name,))
            await db.commit()

    @staticmethod
    async def list_enabled() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM mcp_servers WHERE enabled = 1")
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                if row.get("env_vars"):
                    row["env_vars"] = json.loads(row["env_vars"])
            return rows

    @staticmethod
    async def upsert(
        name: str, transport: str, command_or_url: str, env_vars: dict | None = None, timeout: int = 30
    ) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO mcp_servers (name, transport, command_or_url, env_vars, timeout, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET transport=?, command_or_url=?, env_vars=?, timeout=?, updated_at=?",
                (
                    name,
                    transport,
                    command_or_url,
                    json.dumps(env_vars) if env_vars else None,
                    timeout,
                    _now(),
                    transport,
                    command_or_url,
                    json.dumps(env_vars) if env_vars else None,
                    timeout,
                    _now(),
                ),
            )
            await db.commit()

    @staticmethod
    async def set_enabled(name: str, enabled: bool) -> None:
        async with get_db_write() as db:
            await db.execute(
                "UPDATE mcp_servers SET enabled = ?, updated_at = ? WHERE name = ?",
                (1 if enabled else 0, _now(), name),
            )
            await db.commit()


class AgentMcpToolsRepository:
    """CRUD for MCP tool assignments to agents (built-in and custom)."""

    @staticmethod
    async def get_tools(agent_id: str) -> list[dict[str, str]]:
        """Return list of {server_name, tool_name} for an agent."""
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT server_name, tool_name FROM agent_mcp_tools WHERE agent_id = ?",
                (agent_id,),
            )
            return [
                {"server_name": row["server_name"], "tool_name": row["tool_name"]} for row in await cursor.fetchall()
            ]

    @staticmethod
    async def assign_tool(agent_id: str, server_name: str, tool_name: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT OR IGNORE INTO agent_mcp_tools (agent_id, server_name, tool_name) VALUES (?, ?, ?)",
                (agent_id, server_name, tool_name),
            )
            await db.commit()

    @staticmethod
    async def unassign_tool(agent_id: str, server_name: str, tool_name: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "DELETE FROM agent_mcp_tools WHERE agent_id = ? AND server_name = ? AND tool_name = ?",
                (agent_id, server_name, tool_name),
            )
            await db.commit()

    @staticmethod
    async def get_all_assignments() -> list[dict[str, str]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT agent_id, server_name, tool_name FROM agent_mcp_tools")
            return [dict(row) for row in await cursor.fetchall()]


class EntityVisibilityRepository:
    """CRUD for per-agent entity visibility rules."""

    @staticmethod
    async def get_rules(agent_id: str) -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT rule_type, rule_value FROM entity_visibility_rules WHERE agent_id = ?",
                (agent_id,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def set_rules(agent_id: str, rules: list[dict[str, str]]) -> None:
        async with get_db_write() as db:
            await db.execute(
                "DELETE FROM entity_visibility_rules WHERE agent_id = ?",
                (agent_id,),
            )
            for rule in rules:
                await db.execute(
                    "INSERT INTO entity_visibility_rules (agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
                    (agent_id, rule["rule_type"], rule["rule_value"]),
                )
            await db.commit()

    @staticmethod
    async def add_rule(agent_id: str, rule_type: str, rule_value: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT OR IGNORE INTO entity_visibility_rules (agent_id, rule_type, rule_value) VALUES (?, ?, ?)",
                (agent_id, rule_type, rule_value),
            )
            await db.commit()

    @staticmethod
    async def remove_rule(agent_id: str, rule_type: str, rule_value: str) -> None:
        async with get_db_write() as db:
            await db.execute(
                "DELETE FROM entity_visibility_rules WHERE agent_id = ? AND rule_type = ? AND rule_value = ?",
                (agent_id, rule_type, rule_value),
            )
            await db.commit()

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT agent_id, rule_type, rule_value FROM entity_visibility_rules")
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def list_domain_include_rules() -> list[dict[str, Any]]:
        """Return all domain_include rules: [{agent_id, rule_value}]."""
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT agent_id, rule_value FROM entity_visibility_rules WHERE rule_type = 'domain_include'"
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def list_device_class_include_rules() -> list[dict[str, Any]]:
        """Return all device_class_include rules: [{agent_id, rule_value}]."""
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT agent_id, rule_value FROM entity_visibility_rules WHERE rule_type = 'device_class_include'"
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def set_domain_agents(domain: str, agent_ids: list[str]) -> None:
        """Set which agents have domain_include for a given domain."""
        async with get_db_write() as db:
            await db.execute(
                "DELETE FROM entity_visibility_rules WHERE rule_type = 'domain_include' AND rule_value = ?",
                (domain,),
            )
            for agent_id in agent_ids:
                await db.execute(
                    "INSERT OR IGNORE INTO entity_visibility_rules "
                    "(agent_id, rule_type, rule_value) VALUES (?, 'domain_include', ?)",
                    (agent_id, domain),
                )
            await db.commit()

    @staticmethod
    async def set_device_class_agents(device_class: str, agent_ids: list[str]) -> None:
        """Set which agents have device_class_include for a given device_class.

        Also ensures each agent has domain_include:sensor so the matcher
        can reach the device_class filter stage.
        """
        async with get_db_write() as db:
            await db.execute(
                "DELETE FROM entity_visibility_rules WHERE rule_type = 'device_class_include' AND rule_value = ?",
                (device_class,),
            )
            for agent_id in agent_ids:
                await db.execute(
                    "INSERT OR IGNORE INTO entity_visibility_rules "
                    "(agent_id, rule_type, rule_value) VALUES (?, 'device_class_include', ?)",
                    (agent_id, device_class),
                )
                # Ensure agent has domain_include:sensor so matcher passes domain filter
                await db.execute(
                    "INSERT OR IGNORE INTO entity_visibility_rules "
                    "(agent_id, rule_type, rule_value) VALUES (?, 'domain_include', 'sensor')",
                    (agent_id,),
                )
            await db.commit()


class PluginRepository:
    """CRUD for plugin metadata."""

    @staticmethod
    async def get(name: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM plugins WHERE name = ?", (name,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM plugins")
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def upsert(name: str, file_path: str, **kwargs: Any) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO plugins (name, file_path, enabled, version, description, loaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET file_path=?, enabled=?, version=?, description=?, loaded_at=?",
                (
                    name,
                    file_path,
                    kwargs.get("enabled", 1),
                    kwargs.get("version"),
                    kwargs.get("description"),
                    _now(),
                    file_path,
                    kwargs.get("enabled", 1),
                    kwargs.get("version"),
                    kwargs.get("description"),
                    _now(),
                ),
            )
            await db.commit()


class ConversationRepository:
    """CRUD for conversation history."""

    @staticmethod
    async def insert(
        conversation_id: str,
        user_text: str,
        agent_id: str | None = None,
        response_text: str | None = None,
        action_executed: str | None = None,
        cache_hit: str | None = None,
        latency_ms: float | None = None,
    ) -> int:
        async with get_db_write() as db:
            cursor = await db.execute(
                "INSERT INTO conversations "
                "(conversation_id, user_text, agent_id, response_text, "
                "action_executed, cache_hit, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, user_text, agent_id, response_text, action_executed, cache_hit, latency_ms),
            )
            await db.commit()
            return cursor.lastrowid

    @staticmethod
    async def list_recent(limit: int = 50) -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def get_by_conversation_id(conversation_id: str) -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def search(
        agent_id: str | None = None,
        search_text: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if search_text:
            conditions.append("(user_text LIKE ? OR response_text LIKE ?)")
            like = f"%{search_text}%"
            params.extend([like, like])
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        offset = (page - 1) * per_page
        params.extend([per_page, offset])

        async with get_db_read() as db:
            cursor = await db.execute(
                f"SELECT * FROM conversations {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params,
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def count(
        agent_id: str | None = None,
        search_text: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if search_text:
            conditions.append("(user_text LIKE ? OR response_text LIKE ?)")
            like = f"%{search_text}%"
            params.extend([like, like])
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        async with get_db_read() as db:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM conversations {where}",
                params,
            )
            row = await cursor.fetchone()
            return row[0]


class AnalyticsRepository:
    """CRUD for analytics events."""

    @staticmethod
    async def insert(event_type: str, agent_id: str | None = None, data: dict | None = None) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO analytics (event_type, agent_id, data) VALUES (?, ?, ?)",
                (event_type, agent_id, json.dumps(data) if data else None),
            )
            await db.commit()

    @staticmethod
    async def query_by_range(
        event_type: str | None = None, start: str | None = None, end: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if start:
            conditions.append("created_at >= ?")
            params.append(start)
        if end:
            conditions.append("created_at <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        async with get_db_read() as db:
            cursor = await db.execute(
                f"SELECT * FROM analytics {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                if row.get("data"):
                    row["data"] = json.loads(row["data"])
            return rows


class EntityMatchingConfigRepository:
    """CRUD for entity matching configuration."""

    @staticmethod
    async def get(key: str) -> str | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT value FROM entity_matching_config WHERE key = ?", (key,))
            row = await cursor.fetchone()
            return row[0] if row else None

    @staticmethod
    async def set(key: str, value: str, description: str | None = None) -> None:
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO entity_matching_config (key, value, description, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                (key, value, description, _now(), value, _now()),
            )
            await db.commit()

    @staticmethod
    async def get_all() -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT key, value, description FROM entity_matching_config")
            return [dict(row) for row in await cursor.fetchall()]


class TraceSpanRepository:
    """CRUD for trace span data."""

    @staticmethod
    async def insert(
        trace_id: str,
        span_name: str,
        start_time: str,
        duration_ms: float,
        agent_id: str | None = None,
        parent_span: str | None = None,
        status: str = "ok",
        metadata: dict | None = None,
        end_time: str | None = None,
    ) -> int:
        async with get_db_write() as db:
            cursor = await db.execute(
                "INSERT INTO trace_spans "
                "(trace_id, span_name, agent_id, parent_span, start_time, "
                "end_time, duration_ms, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    span_name,
                    agent_id,
                    parent_span,
                    start_time,
                    end_time,
                    duration_ms,
                    status,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    @staticmethod
    async def insert_batch(spans: list[dict[str, Any]]) -> None:
        async with get_db_write() as db:
            for span in spans:
                meta = dict(span.get("metadata") or {})
                if span.get("span_id"):
                    meta["span_id"] = span["span_id"]
                await db.execute(
                    "INSERT INTO trace_spans "
                    "(trace_id, span_name, agent_id, parent_span, start_time, "
                    "end_time, duration_ms, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        span["trace_id"],
                        span["span_name"],
                        span.get("agent_id"),
                        span.get("parent_span"),
                        span["start_time"],
                        span.get("end_time"),
                        span["duration_ms"],
                        span.get("status", "ok"),
                        json.dumps(meta) if meta else None,
                    ),
                )
            await db.commit()

    @staticmethod
    async def list_traces(page: int = 1, per_page: int = 50) -> list[dict[str, Any]]:
        offset = (page - 1) * per_page
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT trace_id, MIN(start_time) as start_time, "
                "COUNT(*) as span_count, "
                "SUM(duration_ms) as total_duration_ms, "
                "GROUP_CONCAT(DISTINCT agent_id) as agents "
                "FROM trace_spans GROUP BY trace_id "
                "ORDER BY start_time DESC LIMIT ? OFFSET ?",
                (per_page, offset),
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def get_trace_spans(trace_id: str) -> list[dict[str, Any]]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT * FROM trace_spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,),
            )
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                if row.get("metadata"):
                    row["metadata"] = json.loads(row["metadata"])
            return rows

    @staticmethod
    async def count_traces() -> int:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT COUNT(DISTINCT trace_id) FROM trace_spans")
            row = await cursor.fetchone()
            return row[0]

    @staticmethod
    async def cleanup_old(days: int = 30) -> int:
        async with get_db_write() as db:
            cursor = await db.execute(
                "DELETE FROM trace_spans WHERE created_at < datetime('now', ?)",
                (f"-{days} days",),
            )
            await db.commit()
            return cursor.rowcount


class TraceSummaryRepository:
    """CRUD for trace summary records."""

    @staticmethod
    async def create(data: dict[str, Any]) -> None:
        agents = data.get("agents")
        if isinstance(agents, (list, dict)):
            agents = json.dumps(agents)
        agent_instructions = data.get("agent_instructions")
        if isinstance(agent_instructions, (list, dict)):
            agent_instructions = json.dumps(agent_instructions)
        conversation_turns = data.get("conversation_turns")
        if isinstance(conversation_turns, list):
            conversation_turns = json.dumps(conversation_turns)
        async with get_db_write() as db:
            await db.execute(
                "INSERT INTO trace_summary "
                "(trace_id, conversation_id, user_input, final_response, "
                "agents, total_duration_ms, label, source, routing_agent, "
                "routing_confidence, routing_duration_ms, routing_reasoning, "
                "agent_instructions, conversation_turns, "
                "device_id, area_id, device_name, area_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data.get("trace_id"),
                    data.get("conversation_id"),
                    data.get("user_input"),
                    data.get("final_response"),
                    agents,
                    data.get("total_duration_ms"),
                    data.get("label"),
                    data.get("source"),
                    data.get("routing_agent"),
                    data.get("routing_confidence"),
                    data.get("routing_duration_ms"),
                    data.get("routing_reasoning"),
                    agent_instructions,
                    conversation_turns,
                    data.get("device_id"),
                    data.get("area_id"),
                    data.get("device_name"),
                    data.get("area_name"),
                ),
            )
            await db.commit()

    @staticmethod
    async def list_filtered(
        search: str | None = None,
        agent: str | None = None,
        label: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if search:
            conditions.append("user_input LIKE ?")
            params.append(f"%{search}%")
        if agent:
            conditions.append("routing_agent = ?")
            params.append(agent)
        if label:
            conditions.append("label = ?")
            params.append(label)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        offset = (page - 1) * per_page
        params.extend([per_page, offset])

        async with get_db_read() as db:
            cursor = await db.execute(
                f"SELECT * FROM trace_summary {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params,
            )
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                if row.get("agents"):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        row["agents"] = json.loads(row["agents"])
                if row.get("agent_instructions"):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        row["agent_instructions"] = json.loads(row["agent_instructions"])
            return rows

    @staticmethod
    async def count_filtered(
        search: str | None = None,
        agent: str | None = None,
        label: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        if search:
            conditions.append("user_input LIKE ?")
            params.append(f"%{search}%")
        if agent:
            conditions.append("routing_agent = ?")
            params.append(agent)
        if label:
            conditions.append("label = ?")
            params.append(label)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        async with get_db_read() as db:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM trace_summary {where}",
                params,
            )
            row = await cursor.fetchone()
            return row[0]

    @staticmethod
    async def get(trace_id: str) -> dict[str, Any] | None:
        async with get_db_read() as db:
            cursor = await db.execute("SELECT * FROM trace_summary WHERE trace_id = ?", (trace_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            result = dict(row)
            if result.get("agents"):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    result["agents"] = json.loads(result["agents"])
            if result.get("agent_instructions"):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    result["agent_instructions"] = json.loads(result["agent_instructions"])
            if result.get("conversation_turns"):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    result["conversation_turns"] = json.loads(result["conversation_turns"])
            return result

    @staticmethod
    async def update_label(trace_id: str, label: str | None) -> None:
        async with get_db_write() as db:
            await db.execute(
                "UPDATE trace_summary SET label = ? WHERE trace_id = ?",
                (label, trace_id),
            )
            await db.commit()

    @staticmethod
    async def list_labels() -> list[str]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT DISTINCT label FROM trace_summary WHERE label IS NOT NULL AND label != '' ORDER BY label"
            )
            return [row[0] for row in await cursor.fetchall()]

    @staticmethod
    async def list_agents() -> list[str]:
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT DISTINCT routing_agent FROM trace_summary "
                "WHERE routing_agent IS NOT NULL AND routing_agent != '' "
                "ORDER BY routing_agent"
            )
            agents = [row[0] for row in await cursor.fetchall()]
            if "orchestrator" not in agents and agents:
                agents.insert(0, "orchestrator")
            return agents

    @staticmethod
    async def export_filtered(
        search: str | None = None,
        agent: str | None = None,
        label: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if search:
            conditions.append("user_input LIKE ?")
            params.append(f"%{search}%")
        if agent:
            conditions.append("routing_agent = ?")
            params.append(agent)
        if label:
            conditions.append("label = ?")
            params.append(label)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(10000)

        async with get_db_read() as db:
            cursor = await db.execute(
                f"SELECT * FROM trace_summary {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                if row.get("agents"):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        row["agents"] = json.loads(row["agents"])
                if row.get("agent_instructions"):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        row["agent_instructions"] = json.loads(row["agent_instructions"])
            return rows

    @staticmethod
    async def cleanup_old(days: int = 30) -> int:
        async with get_db_write() as db:
            cursor = await db.execute(
                "DELETE FROM trace_summary WHERE created_at < datetime('now', ?)",
                (f"-{days} days",),
            )
            await db.commit()
            return cursor.rowcount

    @staticmethod
    async def update_duration(trace_id: str, duration_ms: float) -> None:
        async with get_db_write() as db:
            await db.execute(
                "UPDATE trace_summary SET total_duration_ms = ? WHERE trace_id = ?",
                (duration_ms, trace_id),
            )
            await db.commit()


def _normalize_device_name(name: str) -> str:
    """Normalize a device display name for fuzzy comparison."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", name.lower())).strip()


class SendDeviceMappingRepository:
    """CRUD for send device name-to-service mappings."""

    @staticmethod
    async def list_all() -> list[dict[str, Any]]:
        """Return all device mappings."""
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT id, display_name, device_type, ha_service_target, created_at "
                "FROM send_device_mappings ORDER BY display_name"
            )
            return [dict(row) for row in await cursor.fetchall()]

    @staticmethod
    async def get(mapping_id: int) -> dict[str, Any] | None:
        """Get a single mapping by ID."""
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT id, display_name, device_type, ha_service_target, created_at "
                "FROM send_device_mappings WHERE id = ?",
                (mapping_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def find_by_name(name: str) -> dict[str, Any] | None:
        """Find a mapping by display_name (case-insensitive, with normalized fallback)."""
        async with get_db_read() as db:
            cursor = await db.execute(
                "SELECT id, display_name, device_type, ha_service_target, created_at "
                "FROM send_device_mappings WHERE display_name = ? COLLATE NOCASE",
                (name.strip(),),
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
            # Fallback: normalized comparison (handles apostrophes, hyphens, etc.)
            normalized_input = _normalize_device_name(name)
            if not normalized_input:
                return None
            cursor = await db.execute(
                "SELECT id, display_name, device_type, ha_service_target, created_at FROM send_device_mappings"
            )
            for row in await cursor.fetchall():
                if _normalize_device_name(row["display_name"]) == normalized_input:
                    return dict(row)
            return None

    @staticmethod
    async def create(display_name: str, device_type: str, ha_service_target: str) -> int:
        """Insert a new mapping. Returns the new row ID."""
        async with get_db_write() as db:
            cursor = await db.execute(
                "INSERT INTO send_device_mappings (display_name, device_type, ha_service_target, created_at) "
                "VALUES (?, ?, ?, ?)",
                (display_name.strip(), device_type, ha_service_target, _now()),
            )
            await db.commit()
            return cursor.lastrowid

    @staticmethod
    async def update(mapping_id: int, **kwargs: Any) -> bool:
        """Update fields of an existing mapping. Returns True if row existed."""
        allowed = {"display_name", "device_type", "ha_service_target"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = [*list(fields.values()), mapping_id]
        async with get_db_write() as db:
            cursor = await db.execute(
                f"UPDATE send_device_mappings SET {set_clause} WHERE id = ?",
                values,
            )
            await db.commit()
            return cursor.rowcount > 0

    @staticmethod
    async def delete(mapping_id: int) -> bool:
        """Delete a mapping by ID. Returns True if row existed."""
        async with get_db_write() as db:
            cursor = await db.execute(
                "DELETE FROM send_device_mappings WHERE id = ?",
                (mapping_id,),
            )
            await db.commit()
            return cursor.rowcount > 0
