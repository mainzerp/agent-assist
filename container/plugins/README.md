# HA-AgentHub Plugins

Place Python plugin files (`.py`) in this directory. They are discovered and
loaded automatically when the container starts.

## Quickstart

1. Create a `.py` file in this directory (e.g., `my_plugin.py`)
2. Subclass `BasePlugin` from `app.plugins.base`
3. Implement the required `name` and `version` properties
4. Optionally implement lifecycle hooks: `configure`, `startup`, `ready`, `shutdown`
5. Restart the container

## Minimal Example

```python
from app.plugins.base import BasePlugin, PluginContext


class HelloPlugin(BasePlugin):

    @property
    def name(self) -> str:
        return "hello"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Logs a greeting at startup"

    async def ready(self, ctx: PluginContext) -> None:
        import logging
        logging.getLogger(__name__).info("Hello from plugin!")
```

## Lifecycle Order

1. **configure** -- Read settings, validate configuration
2. **startup** -- Initialize resources (connections, caches)
3. **ready** -- Register agents, subscribe to events (system is fully up)
4. **shutdown** -- Clean up resources (called when container stops)

## Available APIs

Through the `PluginContext` object passed to lifecycle hooks:

- `ctx.agent_registry` -- Register/unregister A2A agents
- `ctx.mcp_registry` -- Access MCP server connections
- `ctx.settings` -- Read/write settings via `SettingsRepository`
- `ctx.event_bus` -- Subscribe to / publish plugin events
- `ctx.add_api_route(path, endpoint, **kwargs)` and
  `ctx.include_router(router, **kwargs)` -- register HTTP routes

Direct access to the FastAPI application instance via `ctx.app` was
removed in 0.13.0. Use `ctx.add_api_route` or `ctx.include_router`
instead; arbitrary `app` mutation is no longer part of the plugin
trust boundary.

## Notes

- Files starting with `_` are ignored
- Each file should contain exactly one `BasePlugin` subclass
- Plugins can be enabled/disabled from the admin dashboard
- Errors in one plugin do not affect other plugins

For full documentation, see `docs/plugin-development.md`.
