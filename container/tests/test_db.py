"""Tests for app.db -- schema creation, seed data, and repository CRUD."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from app.db.repository import (
    AdminAccountRepository,
    AgentConfigRepository,
    AliasRepository,
    CustomAgentRepository,
    McpServerRepository,
    SecretsRepository,
    SettingsRepository,
    SetupStateRepository,
)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestSchemaCreation:

    async def test_all_expected_tables_exist(self, db_repository):
        expected_tables = {
            "schema_version", "settings", "agent_configs", "custom_agents",
            "entity_matching_config", "aliases", "mcp_servers", "secrets",
            "admin_accounts", "setup_state", "entity_visibility_rules",
            "plugins", "conversations", "analytics", "trace_spans",
        }
        async with aiosqlite.connect(str(db_repository)) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            rows = await cursor.fetchall()
            actual_tables = {row[0] for row in rows}
        assert expected_tables.issubset(actual_tables), f"Missing: {expected_tables - actual_tables}"

    async def test_indexes_created(self, db_repository):
        async with aiosqlite.connect(str(db_repository)) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            rows = await cursor.fetchall()
            index_names = {row[0] for row in rows}
        assert "idx_settings_category" in index_names
        assert "idx_aliases_entity_id" in index_names

    async def test_schema_version_seeded(self, db_repository):
        async with aiosqlite.connect(str(db_repository)) as db:
            cursor = await db.execute("SELECT version FROM schema_version")
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

class TestSeedData:

    async def test_default_settings_populated(self, db_repository):
        all_settings = await SettingsRepository.get_all()
        keys = {s["key"] for s in all_settings}
        assert "cache.routing.threshold" in keys
        assert "embedding.provider" in keys
        assert "a2a.default_timeout" in keys

    async def test_default_agent_configs_populated(self, db_repository):
        agents = await AgentConfigRepository.list_all()
        agent_ids = {a["agent_id"] for a in agents}
        assert "orchestrator" in agent_ids
        assert "light-agent" in agent_ids
        assert "general-agent" in agent_ids

    async def test_default_entity_matching_weights(self, db_repository):
        async with aiosqlite.connect(str(db_repository)) as db:
            cursor = await db.execute("SELECT key, value FROM entity_matching_config")
            rows = await cursor.fetchall()
        weights = {row[0]: row[1] for row in rows}
        assert "weight.levenshtein" in weights
        assert "weight.embedding" in weights
        assert float(weights["weight.embedding"]) == 0.30

    async def test_setup_steps_seeded(self, db_repository):
        steps = await SetupStateRepository.get_all_steps()
        step_names = {s["step"] for s in steps}
        assert "admin_password" in step_names
        assert "review_complete" in step_names
        assert len(steps) == 5


# ---------------------------------------------------------------------------
# Repository CRUD -- settings
# ---------------------------------------------------------------------------

class TestSettingsRepository:

    async def test_get_existing_setting(self, db_repository):
        result = await SettingsRepository.get("cache.routing.threshold")
        assert result is not None
        assert result["value"] == "0.92"

    async def test_get_value_existing(self, db_repository):
        val = await SettingsRepository.get_value("cache.routing.threshold")
        assert val == "0.92"

    async def test_get_value_missing_returns_default(self, db_repository):
        val = await SettingsRepository.get_value("nonexistent.key", "fallback")
        assert val == "fallback"

    async def test_get_missing_key_returns_none(self, db_repository):
        result = await SettingsRepository.get("does.not.exist")
        assert result is None

    async def test_set_new_setting(self, db_repository):
        await SettingsRepository.set("test.key", "test_value", category="test")
        val = await SettingsRepository.get_value("test.key")
        assert val == "test_value"

    async def test_set_overwrites_existing(self, db_repository):
        await SettingsRepository.set("cache.routing.threshold", "0.85")
        val = await SettingsRepository.get_value("cache.routing.threshold")
        assert val == "0.85"

    async def test_get_by_category(self, db_repository):
        results = await SettingsRepository.get_by_category("cache")
        assert len(results) > 0
        assert all(r["key"].startswith("cache.") for r in results)

    async def test_get_all_returns_many(self, db_repository):
        results = await SettingsRepository.get_all()
        assert len(results) > 10


# ---------------------------------------------------------------------------
# Repository CRUD -- agent_configs
# ---------------------------------------------------------------------------

class TestAgentConfigRepository:

    async def test_get_existing(self, db_repository):
        cfg = await AgentConfigRepository.get("light-agent")
        assert cfg is not None
        assert cfg["agent_id"] == "light-agent"
        assert cfg["enabled"] == 1

    async def test_get_missing_returns_none(self, db_repository):
        cfg = await AgentConfigRepository.get("nonexistent-agent")
        assert cfg is None

    async def test_list_all(self, db_repository):
        agents = await AgentConfigRepository.list_all()
        assert len(agents) >= 11

    async def test_list_enabled(self, db_repository):
        enabled = await AgentConfigRepository.list_enabled()
        for agent in enabled:
            assert agent["enabled"] == 1

    async def test_upsert_update_existing(self, db_repository):
        await AgentConfigRepository.upsert("light-agent", enabled=0)
        cfg = await AgentConfigRepository.get("light-agent")
        assert cfg["enabled"] == 0

    async def test_upsert_create_new(self, db_repository):
        await AgentConfigRepository.upsert("new-test-agent", enabled=1, description="Test")
        cfg = await AgentConfigRepository.get("new-test-agent")
        assert cfg is not None
        assert cfg["description"] == "Test"


# ---------------------------------------------------------------------------
# Repository CRUD -- custom_agents
# ---------------------------------------------------------------------------

class TestCustomAgentRepository:

    async def test_create_and_get(self, db_repository):
        await CustomAgentRepository.create(
            "test-custom",
            system_prompt="You are a test agent",
            description="Test custom agent",
            mcp_tools=["tool1", "tool2"],
            intent_patterns=["pattern_a"],
        )
        agent = await CustomAgentRepository.get("test-custom")
        assert agent is not None
        assert agent["system_prompt"] == "You are a test agent"
        assert agent["mcp_tools"] == ["tool1", "tool2"]
        assert agent["intent_patterns"] == ["pattern_a"]

    async def test_list_all(self, db_repository):
        await CustomAgentRepository.create("ca-1", system_prompt="p1")
        await CustomAgentRepository.create("ca-2", system_prompt="p2")
        agents = await CustomAgentRepository.list_all()
        names = {a["name"] for a in agents}
        assert "ca-1" in names
        assert "ca-2" in names

    async def test_list_enabled(self, db_repository):
        await CustomAgentRepository.create("ca-en", system_prompt="p", enabled=1)
        await CustomAgentRepository.create("ca-dis", system_prompt="p", enabled=0)
        enabled = await CustomAgentRepository.list_enabled()
        names = {a["name"] for a in enabled}
        assert "ca-en" in names
        assert "ca-dis" not in names

    async def test_update(self, db_repository):
        await CustomAgentRepository.create("ca-upd", system_prompt="old")
        await CustomAgentRepository.update("ca-upd", system_prompt="new", description="updated")
        agent = await CustomAgentRepository.get("ca-upd")
        assert agent["system_prompt"] == "new"
        assert agent["description"] == "updated"

    async def test_delete(self, db_repository):
        await CustomAgentRepository.create("ca-del", system_prompt="temp")
        await CustomAgentRepository.delete("ca-del")
        agent = await CustomAgentRepository.get("ca-del")
        assert agent is None

    async def test_json_fields_serialize_correctly(self, db_repository):
        await CustomAgentRepository.create(
            "ca-json",
            system_prompt="p",
            entity_visibility={"domains": ["light"]},
        )
        agent = await CustomAgentRepository.get("ca-json")
        assert isinstance(agent["entity_visibility"], dict)
        assert agent["entity_visibility"]["domains"] == ["light"]


# ---------------------------------------------------------------------------
# Repository CRUD -- aliases
# ---------------------------------------------------------------------------

class TestAliasRepository:

    async def test_set_and_get(self, db_repository):
        await AliasRepository.set("nightstand lamp", "light.bedroom_nightstand")
        result = await AliasRepository.get("nightstand lamp")
        assert result == "light.bedroom_nightstand"

    async def test_get_missing_returns_none(self, db_repository):
        result = await AliasRepository.get("nonexistent_alias")
        assert result is None

    async def test_delete(self, db_repository):
        await AliasRepository.set("temp_alias", "light.temp")
        await AliasRepository.delete("temp_alias")
        result = await AliasRepository.get("temp_alias")
        assert result is None

    async def test_list_all(self, db_repository):
        await AliasRepository.set("alias_a", "light.a")
        await AliasRepository.set("alias_b", "light.b")
        all_aliases = await AliasRepository.list_all()
        alias_keys = {a["alias"] for a in all_aliases}
        assert "alias_a" in alias_keys
        assert "alias_b" in alias_keys

    async def test_upsert_overwrite(self, db_repository):
        await AliasRepository.set("dup", "light.old")
        await AliasRepository.set("dup", "light.new")
        result = await AliasRepository.get("dup")
        assert result == "light.new"


# ---------------------------------------------------------------------------
# Repository CRUD -- mcp_servers
# ---------------------------------------------------------------------------

class TestMcpServerRepository:

    async def test_create_and_get(self, db_repository):
        await McpServerRepository.create("test-mcp", "stdio", "python mcp_server.py")
        server = await McpServerRepository.get("test-mcp")
        assert server is not None
        assert server["transport"] == "stdio"

    async def test_create_with_env_vars(self, db_repository):
        await McpServerRepository.create(
            "mcp-env", "http", "http://localhost:8000",
            env_vars={"API_KEY": "secret123"},
        )
        server = await McpServerRepository.get("mcp-env")
        assert server["env_vars"] == {"API_KEY": "secret123"}

    async def test_delete(self, db_repository):
        await McpServerRepository.create("mcp-del", "stdio", "cmd")
        await McpServerRepository.delete("mcp-del")
        server = await McpServerRepository.get("mcp-del")
        assert server is None

    async def test_list_all(self, db_repository):
        await McpServerRepository.create("mcp-a", "stdio", "a")
        await McpServerRepository.create("mcp-b", "http", "b")
        servers = await McpServerRepository.list_all()
        names = {s["name"] for s in servers}
        assert "mcp-a" in names
        assert "mcp-b" in names

    async def test_list_enabled(self, db_repository):
        await McpServerRepository.create("mcp-on", "stdio", "cmd")
        servers = await McpServerRepository.list_enabled()
        names = {s["name"] for s in servers}
        assert "mcp-on" in names


# ---------------------------------------------------------------------------
# Repository CRUD -- secrets
# ---------------------------------------------------------------------------

class TestSecretsRepository:

    async def test_store_and_retrieve(self, db_repository):
        encrypted = b"encrypted_secret_data"
        await SecretsRepository.set("ha_token", encrypted)
        result = await SecretsRepository.get("ha_token")
        assert result == encrypted

    async def test_stored_value_is_bytes(self, db_repository):
        await SecretsRepository.set("test_secret", b"\x00\x01\x02")
        result = await SecretsRepository.get("test_secret")
        assert isinstance(result, bytes)

    async def test_get_missing_returns_none(self, db_repository):
        result = await SecretsRepository.get("not_a_key")
        assert result is None

    async def test_delete(self, db_repository):
        await SecretsRepository.set("del_key", b"data")
        await SecretsRepository.delete("del_key")
        result = await SecretsRepository.get("del_key")
        assert result is None

    async def test_list_keys(self, db_repository):
        await SecretsRepository.set("k1", b"v1")
        await SecretsRepository.set("k2", b"v2")
        keys = await SecretsRepository.list_keys()
        assert "k1" in keys
        assert "k2" in keys


# ---------------------------------------------------------------------------
# Repository CRUD -- admin_accounts
# ---------------------------------------------------------------------------

class TestAdminAccountRepository:

    async def test_create_and_get(self, db_repository):
        await AdminAccountRepository.create("admin", "$2b$12$fakebcrypthash")
        account = await AdminAccountRepository.get("admin")
        assert account is not None
        assert account["username"] == "admin"
        assert account["password_hash"] == "$2b$12$fakebcrypthash"

    async def test_get_missing_returns_none(self, db_repository):
        account = await AdminAccountRepository.get("nobody")
        assert account is None

    async def test_password_hash_stored(self, db_repository):
        await AdminAccountRepository.create("user1", "$2b$12$somehash")
        account = await AdminAccountRepository.get("user1")
        assert account["password_hash"].startswith("$2b$12$")

    async def test_update_last_login(self, db_repository):
        await AdminAccountRepository.create("loginuser", "$2b$12$hash")
        await AdminAccountRepository.update_last_login("loginuser")
        account = await AdminAccountRepository.get("loginuser")
        assert account["last_login"] is not None

    async def test_list_all(self, db_repository):
        await AdminAccountRepository.create("u1", "$2b$12$h1")
        await AdminAccountRepository.create("u2", "$2b$12$h2")
        accounts = await AdminAccountRepository.list_all()
        usernames = {a["username"] for a in accounts}
        assert "u1" in usernames
        assert "u2" in usernames

    async def test_duplicate_username_raises(self, db_repository):
        await AdminAccountRepository.create("dupuser", "$2b$12$h")
        with pytest.raises(Exception):
            await AdminAccountRepository.create("dupuser", "$2b$12$h2")


# ---------------------------------------------------------------------------
# Repository CRUD -- setup_state
# ---------------------------------------------------------------------------

class TestSetupStateRepository:

    async def test_get_step(self, db_repository):
        step = await SetupStateRepository.get_step("admin_password")
        assert step is not None
        assert step["completed"] == 0

    async def test_set_step_completed(self, db_repository):
        await SetupStateRepository.set_step_completed("admin_password")
        step = await SetupStateRepository.get_step("admin_password")
        assert step["completed"] == 1
        assert step["completed_at"] is not None

    async def test_is_complete_initially_false(self, db_repository):
        result = await SetupStateRepository.is_complete()
        assert result is False

    async def test_is_complete_after_all_completed(self, db_repository):
        steps = await SetupStateRepository.get_all_steps()
        for s in steps:
            await SetupStateRepository.set_step_completed(s["step"])
        result = await SetupStateRepository.is_complete()
        assert result is True

    async def test_get_all_steps(self, db_repository):
        steps = await SetupStateRepository.get_all_steps()
        assert len(steps) == 5
