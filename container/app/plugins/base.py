"""Base plugin class and context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from app.a2a.registry import AgentRegistry
    from app.mcp.registry import MCPServerRegistry
    from app.db.repository import SettingsRepository


class PluginContext:
    """Context object passed to plugin lifecycle hooks.

    Provides a restricted API surface. Plugins should NOT have
    unrestricted access to the full FastAPI application.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        mcp_registry: MCPServerRegistry,
        settings_repo: type[SettingsRepository],
        app: FastAPI,
    ) -> None:
        self.agent_registry = agent_registry
        self.mcp_registry = mcp_registry
        self.settings = settings_repo
        self._app = app
        self.event_bus = None  # Set by PluginLoader after construction

    def add_api_route(self, path: str, endpoint, **kwargs):
        """Add an API route to the application (restricted interface)."""
        self._app.add_api_route(path, endpoint, **kwargs)

    def include_router(self, router, **kwargs):
        """Include an APIRouter in the application (restricted interface)."""
        self._app.include_router(router, **kwargs)

    @property
    def app(self):
        """Direct app access has been removed.

        Use add_api_route() or include_router() instead.
        """
        raise AttributeError(
            "PluginContext.app has been removed. "
            "Use add_api_route() or include_router() instead."
        )


class BasePlugin(ABC):
    """Abstract base class for all plugins.

    Subclasses must implement ``name`` and ``version`` properties.
    Lifecycle hooks (configure, startup, ready, shutdown) are optional
    and default to no-ops.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the plugin."""
        return ""

    async def configure(self, ctx: PluginContext) -> None:
        """Called during the CONFIGURE phase. Read settings here."""

    async def startup(self, ctx: PluginContext) -> None:
        """Called during the STARTUP phase. Initialize resources here."""

    async def ready(self, ctx: PluginContext) -> None:
        """Called when all agents are registered and the system is ready."""

    async def shutdown(self) -> None:
        """Called at system shutdown. Clean up resources here."""
