"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.schema import init_db
from app.db.repository import SetupStateRepository, AgentConfigRepository, SettingsRepository
from app.middleware.auth import apply_auth_dependencies, SetupRedirectMiddleware
from app.middleware.tracing import TracingMiddleware
from app.a2a.registry import registry
from app.a2a.transport import InProcessTransport
from app.a2a.dispatcher import Dispatcher
from app.agents.orchestrator import OrchestratorAgent
from app.agents.general import GeneralAgent
from app.agents.light import LightAgent
from app.agents.music import MusicAgent
from app.agents.timer import TimerAgent
from app.agents.climate import ClimateAgent
from app.agents.media import MediaAgent
from app.agents.scene import SceneAgent
from app.agents.automation import AutomationAgent
from app.agents.security import SecurityAgent
from app.agents.rewrite import RewriteAgent
from app.agents.custom_loader import CustomAgentLoader
from app.presence.detector import PresenceDetector
from app.api.routes import conversation as conversation_routes
from app.api.routes import admin as admin_routes
from app.api.routes import health as health_routes
from app.api.routes import dashboard_api as dashboard_api_routes
from app.dashboard.routes import router as dashboard_router
from app.setup.routes import router as setup_router
from app.models.entity_index import EntityIndexEntry

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure structured logging based on settings."""
    log_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=log_format,
        force=True,
    )


def _parse_ha_states(states: list[dict[str, Any]]) -> list[EntityIndexEntry]:
    """Parse HA GET /api/states response into EntityIndexEntry list."""
    entries = []
    for state in states:
        entity_id = state.get("entity_id", "")
        attrs = state.get("attributes", {})
        domain = entity_id.split(".")[0] if "." in entity_id else ""
        entries.append(EntityIndexEntry(
            entity_id=entity_id,
            friendly_name=attrs.get("friendly_name", ""),
            domain=domain,
            area=attrs.get("area_id"),
            device_class=attrs.get("device_class"),
            aliases=[],
        ))
    return entries


async def _periodic_entity_sync(app: FastAPI) -> None:
    """Periodically sync entity index with Home Assistant state."""
    while True:
        try:
            raw = await SettingsRepository.get_value(
                "entity_sync.interval_minutes", "30"
            )
            interval_minutes = int(raw)
        except (TypeError, ValueError):
            interval_minutes = 30

        if interval_minutes <= 0:
            # Disabled -- check again in 5 minutes in case setting changes
            await asyncio.sleep(300)
            continue

        await asyncio.sleep(interval_minutes * 60)

        try:
            ha_client = app.state.ha_client
            entity_index = app.state.entity_index
            if not ha_client or not entity_index:
                continue

            states = await ha_client.get_states()
            entities = _parse_ha_states(states)
            result = entity_index.sync(entities)
            logger.info(
                "Periodic entity sync: +%d ~%d -%d =%d",
                result["added"], result["updated"],
                result["removed"], result["unchanged"],
            )
        except Exception:
            logger.warning("Periodic entity sync failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # --- Startup ---
    _configure_logging()
    logger.info("Starting agent-assist container")
    await init_db()

    from app.security.encryption import is_fernet_key_present
    if is_fernet_key_present():
        logger.warning(
            "IMPORTANT: Back up your Fernet key at /data/.fernet_key. "
            "Loss of this file makes all encrypted secrets (HA token, LLM keys, API key) unrecoverable."
        )

    # Register default sync interval setting if not already set
    existing = await SettingsRepository.get_value("entity_sync.interval_minutes")
    if existing is None:
        await SettingsRepository.set(
            "entity_sync.interval_minutes",
            "30",
            value_type="number",
            category="sync",
            description="Minutes between periodic entity index syncs (0 = disabled)",
        )

    # Check if setup is complete before initializing HA-dependent components
    setup_complete = await SetupStateRepository.is_complete()

    ha_client = None
    entity_index = None
    cache_manager = None
    entity_matcher = None
    alias_resolver = None
    rewrite_agent = None

    if setup_complete:
        from app.cache.embedding import get_embedding_engine
        from app.cache.vector_store import get_vector_store
        from app.cache.cache_manager import CacheManager
        from app.ha_client.rest import HARestClient
        from app.entity.index import EntityIndex
        from app.entity.aliases import AliasResolver
        from app.entity.matcher import EntityMatcher

        # Initialize embedding engine + vector store
        await get_embedding_engine()
        vector_store = await get_vector_store()

        # Initialize HA REST client
        ha_client = HARestClient()
        await ha_client.initialize()

        # Initialize entity index and populate from HA
        entity_index = EntityIndex(vector_store)
        try:
            states = await ha_client.get_states()
            entities = _parse_ha_states(states)
            entity_index.populate(entities)
            logger.info("Entity index populated with %d entities", len(entities))
        except Exception:
            logger.warning("Failed to populate entity index from HA", exc_info=True)

        # Initialize alias resolver
        alias_resolver = AliasResolver()
        await alias_resolver.load()

        # Initialize entity matcher
        entity_matcher = EntityMatcher(entity_index, alias_resolver)
        await entity_matcher.load_config()

        # Initialize rewrite agent and cache manager
        rewrite_agent = RewriteAgent(ha_client=ha_client, entity_index=entity_index)
        cache_manager = CacheManager(vector_store, rewrite_agent=rewrite_agent)
        await cache_manager.initialize()

    # Initialize A2A layer
    transport = InProcessTransport(registry)
    dispatcher = Dispatcher(registry, transport)

    # Inject dispatcher and registry into route modules
    conversation_routes.set_dispatcher(dispatcher)
    dashboard_api_routes.set_chat_dispatcher(dispatcher)
    admin_routes.set_registry(registry)

    # Initialize presence detector (will be fully initialized after WS setup)
    presence_detector = None
    if setup_complete and ha_client:
        presence_detector = PresenceDetector(ha_client)
        await presence_detector.initialize()

    # Register agents with HA client and entity index
    orchestrator_agent = OrchestratorAgent(
        dispatcher=dispatcher,
        registry=registry,
        cache_manager=cache_manager,
        presence_detector=presence_detector,
        ha_client=ha_client,
        entity_index=entity_index,
    )
    await registry.register(orchestrator_agent)

    general_agent = GeneralAgent(ha_client=ha_client, entity_index=entity_index)
    await registry.register(general_agent)

    light_agent = LightAgent(ha_client=ha_client, entity_index=entity_index, entity_matcher=entity_matcher)
    await registry.register(light_agent)

    music_agent = MusicAgent(ha_client=ha_client, entity_index=entity_index, entity_matcher=entity_matcher)
    await registry.register(music_agent)

    # Register Phase 2 agents (only if enabled in agent_configs)
    if setup_complete:
        _PHASE2_AGENTS = [
            ("timer-agent", TimerAgent),
            ("climate-agent", ClimateAgent),
            ("media-agent", MediaAgent),
            ("scene-agent", SceneAgent),
            ("automation-agent", AutomationAgent),
            ("security-agent", SecurityAgent),
        ]
        _PHASE2_AGENTS_WITH_MATCHER = {"climate-agent", "security-agent", "timer-agent", "scene-agent", "automation-agent", "media-agent"}
        for agent_id, agent_cls in _PHASE2_AGENTS:
            config = await AgentConfigRepository.get(agent_id)
            if config and config.get("enabled"):
                if agent_id in _PHASE2_AGENTS_WITH_MATCHER:
                    agent = agent_cls(ha_client=ha_client, entity_index=entity_index, entity_matcher=entity_matcher)
                else:
                    agent = agent_cls(ha_client=ha_client, entity_index=entity_index)
                await registry.register(agent)

        # Register rewrite agent (internal use, not routable by orchestrator)
        await registry.register(rewrite_agent)

    # Load custom agents from DB (Batch 4)
    custom_loader = CustomAgentLoader(registry, ha_client=ha_client, entity_index=entity_index)
    await custom_loader.load_all()

    # Initialize MCP server registry and tool manager (Batch 5A)
    from app.mcp.registry import MCPServerRegistry
    from app.mcp.tools import MCPToolManager

    mcp_registry = MCPServerRegistry()
    mcp_tool_manager = MCPToolManager(mcp_registry)
    if setup_complete:
        try:
            await mcp_registry.load_from_db()
        except Exception:
            logger.warning("Failed to load MCP servers from DB", exc_info=True)

    # Load orchestrator reliability config
    await orchestrator_agent.initialize()

    logger.info("Registered %d agents", len(await registry.list_agents()))

    # Start WebSocket client for real-time entity index refresh (Batch 5B)
    ws_client = None
    ws_task = None
    if setup_complete and entity_index is not None:
        from app.ha_client.websocket import HAWebSocketClient

        ws_client = HAWebSocketClient()

        def on_state_changed(event: dict) -> None:
            """Handle state_changed events for entity index refresh."""
            data = event.get("data", {})
            new_state = data.get("new_state")
            old_state = data.get("old_state")
            entity_id = data.get("entity_id", "")

            if new_state is None and old_state is not None:
                entity_index.remove(entity_id)
            elif new_state is not None:
                attrs = new_state.get("attributes", {})
                domain = entity_id.split(".")[0] if "." in entity_id else ""
                entry = EntityIndexEntry(
                    entity_id=entity_id,
                    friendly_name=attrs.get("friendly_name", ""),
                    domain=domain,
                    area=attrs.get("area_id"),
                    device_class=attrs.get("device_class"),
                    aliases=[],
                )
                entity_index.add(entry)

        ws_client.on_event("state_changed", on_state_changed)

        # Wire presence sensor updates into WebSocket event handler
        if presence_detector:
            def on_state_changed_presence(event: dict) -> None:
                """Handle state_changed events for presence sensor updates."""
                data = event.get("data", {})
                entity_id = data.get("entity_id", "")
                new_state = data.get("new_state")
                if new_state and entity_id.startswith("binary_sensor."):
                    area = new_state.get("attributes", {}).get("area_id")
                    presence_detector.on_sensor_state_change(
                        entity_id, new_state.get("state", "off"), area
                    )

            ws_client.on_event("state_changed", on_state_changed_presence)

        ws_task = asyncio.create_task(ws_client.run())

    # Start periodic entity sync task
    sync_task = None
    if setup_complete and entity_index is not None:
        sync_task = asyncio.create_task(_periodic_entity_sync(app))

    # Store on app.state for access elsewhere if needed
    app.state.startup_time = time.time()
    app.state.registry = registry
    app.state.dispatcher = dispatcher
    app.state.ha_client = ha_client
    app.state.entity_index = entity_index
    app.state.cache_manager = cache_manager
    app.state.entity_matcher = entity_matcher
    app.state.alias_resolver = alias_resolver
    app.state.custom_loader = custom_loader
    app.state.mcp_registry = mcp_registry
    app.state.mcp_tool_manager = mcp_tool_manager
    app.state.ws_client = ws_client
    app.state.presence_detector = presence_detector
    app.state.sync_task = sync_task

    # --- Plugin System (Batch F) ---
    from app.plugins.base import PluginContext
    from app.plugins.loader import PluginLoader
    from app.plugins.hooks import LifecyclePhase

    plugin_context = PluginContext(
        agent_registry=registry,
        mcp_registry=mcp_registry,
        settings_repo=SettingsRepository,
        app=app,
    )
    plugin_dir = str(Path(__file__).resolve().parent.parent / "plugins")
    plugin_loader = PluginLoader(plugin_dir, plugin_context)
    await plugin_loader.discover_and_load()
    await plugin_loader.run_lifecycle(LifecyclePhase.CONFIGURE)
    await plugin_loader.run_lifecycle(LifecyclePhase.STARTUP)
    await plugin_loader.run_lifecycle(LifecyclePhase.READY)
    app.state.plugin_loader = plugin_loader

    logger.info("Startup complete (setup_complete=%s)", setup_complete)
    yield

    # --- Shutdown ---
    logger.info("Shutting down agent-assist container")

    # Plugin shutdown (isolated -- errors must not block remaining cleanup)
    try:
        await plugin_loader.run_lifecycle(LifecyclePhase.SHUTDOWN)
    except Exception:
        logger.warning("Plugin shutdown error (continuing cleanup)", exc_info=True)

    if ws_client:
        await ws_client.disconnect()

    # Cancel background tasks and await them to ensure cleanup completes
    tasks_to_cancel = []
    if ws_task and not ws_task.done():
        ws_task.cancel()
        tasks_to_cancel.append(ws_task)
    if sync_task and not sync_task.done():
        sync_task.cancel()
        tasks_to_cancel.append(sync_task)
    if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    await mcp_registry.disconnect_all()
    if ha_client:
        await ha_client.close()
    from app.db.schema import close_db
    await close_db()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    from app import __version__
    app = FastAPI(
        title="agent-assist",
        version=__version__,
        lifespan=lifespan,
    )

    # Exception handlers
    apply_auth_dependencies(app)

    # Setup redirect middleware (redirects to /setup/ if unconfigured)
    app.add_middleware(SetupRedirectMiddleware)

    # Tracing middleware (trace ID + request logging)
    app.add_middleware(TracingMiddleware)

    # Include routers
    app.include_router(health_routes.router)
    app.include_router(setup_router)
    app.include_router(conversation_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(dashboard_api_routes.router)
    app.include_router(dashboard_router)

    # Batch C routers
    from app.api.routes import conversations_api as conversations_api_routes
    from app.api.routes import cache_api as cache_api_routes
    from app.api.routes import entity_index_api as entity_index_api_routes
    app.include_router(conversations_api_routes.router)
    app.include_router(cache_api_routes.router)
    app.include_router(entity_index_api_routes.router)

    # Batch D routers
    from app.api.routes import analytics_api as analytics_api_routes
    from app.api.routes import traces_api as traces_api_routes
    app.include_router(analytics_api_routes.router)
    app.include_router(traces_api_routes.router)

    # Batch E routers
    from app.api.routes import mcp_api as mcp_api_routes
    from app.api.routes import custom_agents_api as custom_agents_api_routes
    from app.api.routes import entity_visibility_api as entity_visibility_api_routes
    from app.api.routes import presence_api as presence_api_routes
    from app.api.routes import domain_agent_map_api as domain_agent_map_api_routes
    app.include_router(mcp_api_routes.router)
    app.include_router(custom_agents_api_routes.router)
    app.include_router(entity_visibility_api_routes.router)
    app.include_router(entity_visibility_api_routes.entities_router)
    app.include_router(presence_api_routes.router)
    app.include_router(domain_agent_map_api_routes.router)

    # Batch F routers
    from app.api.routes import plugins_api as plugins_api_routes
    app.include_router(plugins_api_routes.router)

    # Try to mount static files (may not exist yet in dev)
    try:
        from pathlib import Path
        static_dir = Path(__file__).parent / "dashboard" / "static"
        if static_dir.is_dir():
            app.mount("/dashboard/static", StaticFiles(directory=str(static_dir)), name="dashboard-static")
    except Exception:
        logger.debug("Static files directory not found, skipping mount")

    return app


app = create_app()
