"""Tests for app.plugins -- base, hooks (EventBus), and loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.plugins.base import BasePlugin, PluginContext
from app.plugins.hooks import EventBus, LifecyclePhase
from app.plugins.loader import PluginLoader


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus:

    async def test_subscribe_and_publish(self):
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("test_event", handler)
        await bus.publish("test_event", {"key": "val"})
        handler.assert_awaited_once_with({"key": "val"})

    async def test_publish_no_subscribers(self):
        bus = EventBus()
        # Should not raise
        await bus.publish("nonexistent_event", None)

    async def test_multiple_handlers_called(self):
        bus = EventBus()
        h1 = AsyncMock()
        h2 = AsyncMock()
        bus.subscribe("event", h1)
        bus.subscribe("event", h2)
        await bus.publish("event", "data")
        h1.assert_awaited_once_with("data")
        h2.assert_awaited_once_with("data")

    async def test_handler_failure_does_not_block_others(self):
        bus = EventBus()
        failing = AsyncMock(side_effect=Exception("boom"))
        ok = AsyncMock()
        bus.subscribe("event", failing)
        bus.subscribe("event", ok)
        await bus.publish("event", None)
        ok.assert_awaited_once()

    def test_clear_removes_all_handlers(self):
        bus = EventBus()
        bus.subscribe("a", AsyncMock())
        bus.subscribe("b", AsyncMock())
        bus.clear()
        assert len(bus._handlers) == 0


# ---------------------------------------------------------------------------
# LifecyclePhase
# ---------------------------------------------------------------------------

class TestLifecyclePhase:

    def test_lifecycle_phases_exist(self):
        assert LifecyclePhase.CONFIGURE.value == "configure"
        assert LifecyclePhase.STARTUP.value == "startup"
        assert LifecyclePhase.READY.value == "ready"
        assert LifecyclePhase.SHUTDOWN.value == "shutdown"


# ---------------------------------------------------------------------------
# BasePlugin
# ---------------------------------------------------------------------------

class TestBasePlugin:

    def test_base_plugin_is_abstract(self):
        with pytest.raises(TypeError):
            BasePlugin()  # type: ignore[abstract]

    def test_concrete_plugin_instantiable(self):
        class MyPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "my-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

        p = MyPlugin()
        assert p.name == "my-plugin"
        assert p.version == "1.0.0"
        assert p.description == ""

    async def test_default_lifecycle_hooks_are_noops(self):
        class NoOpPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "noop"

            @property
            def version(self) -> str:
                return "0.1.0"

        p = NoOpPlugin()
        ctx = MagicMock(spec=PluginContext)
        # These should not raise
        await p.configure(ctx)
        await p.startup(ctx)
        await p.ready(ctx)
        await p.shutdown()


# ---------------------------------------------------------------------------
# PluginContext
# ---------------------------------------------------------------------------

class TestPluginContext:

    def test_context_provides_registries(self):
        agent_reg = MagicMock()
        mcp_reg = MagicMock()
        settings = MagicMock()
        app = MagicMock()
        ctx = PluginContext(
            agent_registry=agent_reg,
            mcp_registry=mcp_reg,
            settings_repo=settings,
            app=app,
        )
        assert ctx.agent_registry is agent_reg
        assert ctx.mcp_registry is mcp_reg
        assert ctx.settings is settings
        assert ctx.app is app


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------

class TestPluginLoader:

    def test_loaded_plugins_empty_initially(self):
        ctx = MagicMock(spec=PluginContext)
        loader = PluginLoader(plugin_dir="/nonexistent", context=ctx)
        assert loader.loaded_plugins == {}

    @patch("app.plugins.loader.PluginRepository")
    async def test_discover_and_load_no_dir(self, mock_repo):
        ctx = MagicMock(spec=PluginContext)
        loader = PluginLoader(plugin_dir="/nonexistent", context=ctx)
        await loader.discover_and_load()
        assert len(loader.loaded_plugins) == 0

    @patch("app.plugins.loader.PluginRepository")
    async def test_run_lifecycle_calls_phase_method(self, mock_repo):
        ctx = MagicMock(spec=PluginContext)
        loader = PluginLoader(plugin_dir="/tmp", context=ctx)

        plugin = MagicMock()
        plugin.configure = AsyncMock()
        loader._loaded = {"test-plugin": plugin}

        await loader.run_lifecycle(LifecyclePhase.CONFIGURE)
        plugin.configure.assert_awaited_once_with(ctx)

    @patch("app.plugins.loader.PluginRepository")
    async def test_run_lifecycle_shutdown_no_ctx_arg(self, mock_repo):
        ctx = MagicMock(spec=PluginContext)
        loader = PluginLoader(plugin_dir="/tmp", context=ctx)

        plugin = MagicMock()
        plugin.shutdown = AsyncMock()
        loader._loaded = {"test-plugin": plugin}

        await loader.run_lifecycle(LifecyclePhase.SHUTDOWN)
        plugin.shutdown.assert_awaited_once_with()

    @patch("app.plugins.loader.PluginRepository")
    async def test_run_lifecycle_isolates_errors(self, mock_repo):
        ctx = MagicMock(spec=PluginContext)
        loader = PluginLoader(plugin_dir="/tmp", context=ctx)

        failing = MagicMock()
        failing.startup = AsyncMock(side_effect=Exception("plugin crash"))
        ok = MagicMock()
        ok.startup = AsyncMock()
        loader._loaded = {"failing": failing, "ok": ok}

        await loader.run_lifecycle(LifecyclePhase.STARTUP)
        ok.startup.assert_awaited_once()

    @patch("app.plugins.loader.PluginRepository")
    async def test_disable_plugin_calls_shutdown(self, mock_repo):
        mock_repo.upsert = AsyncMock()
        ctx = MagicMock(spec=PluginContext)
        loader = PluginLoader(plugin_dir="/tmp", context=ctx)

        plugin = MagicMock()
        plugin.shutdown = AsyncMock()
        loader._loaded = {"test": plugin}

        result = await loader.disable_plugin("test")
        assert result is True
        plugin.shutdown.assert_awaited_once()
        assert "test" not in loader._loaded
