"""Tests for app.config -- AppSettings defaults and env var overrides."""

from __future__ import annotations

import pytest


class TestSettingsDefaults:
    """Verify default values when no env vars are set."""

    def test_default_container_host(self):
        from app.config import Settings
        s = Settings()
        assert s.container_host == "0.0.0.0"

    def test_default_container_port(self):
        from app.config import Settings
        s = Settings()
        assert s.container_port == 8080

    def test_default_log_level(self):
        from app.config import Settings
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_chromadb_persist_dir(self):
        from app.config import Settings
        s = Settings()
        assert s.chromadb_persist_dir == "/data/chromadb"

    def test_default_sqlite_db_path(self):
        from app.config import Settings
        s = Settings()
        assert s.sqlite_db_path == "/data/agent_assist.db"


class TestSettingsEnvOverrides:
    """Verify env var overrides via monkeypatch."""

    def test_override_container_host(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_HOST", "127.0.0.1")
        from app.config import Settings
        s = Settings()
        assert s.container_host == "127.0.0.1"

    def test_override_container_port(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_PORT", "9090")
        from app.config import Settings
        s = Settings()
        assert s.container_port == 9090

    def test_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        from app.config import Settings
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_override_chromadb_persist_dir(self, monkeypatch):
        monkeypatch.setenv("CHROMADB_PERSIST_DIR", "/tmp/chroma")
        from app.config import Settings
        s = Settings()
        assert s.chromadb_persist_dir == "/tmp/chroma"

    def test_override_sqlite_db_path(self, monkeypatch):
        monkeypatch.setenv("SQLITE_DB_PATH", "/tmp/test.db")
        from app.config import Settings
        s = Settings()
        assert s.sqlite_db_path == "/tmp/test.db"


class TestSettingsValidation:
    """Verify type coercion and attribute access."""

    def test_port_accepts_valid_int_string(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_PORT", "3000")
        from app.config import Settings
        s = Settings()
        assert s.container_port == 3000
        assert isinstance(s.container_port, int)

    def test_port_rejects_non_int(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_PORT", "not_a_number")
        from app.config import Settings
        with pytest.raises(Exception):
            Settings()

    def test_settings_attributes_typed(self):
        from app.config import Settings
        s = Settings()
        assert isinstance(s.container_host, str)
        assert isinstance(s.container_port, int)
        assert isinstance(s.log_level, str)

    def test_extra_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv("SOME_UNKNOWN_VAR", "whatever")
        from app.config import Settings
        s = Settings()
        assert not hasattr(s, "some_unknown_var")
