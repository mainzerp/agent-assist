"""Helpers for initializing setup-dependent runtime services in-process."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.agents.automation import AutomationAgent
from app.agents.climate import ClimateAgent
from app.agents.custom_loader import CustomAgentLoader
from app.agents.filler import FillerAgent
from app.agents.general import GeneralAgent
from app.agents.light import LightAgent
from app.agents.media import MediaAgent
from app.agents.music import MusicAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rewrite import RewriteAgent
from app.agents.scene import SceneAgent
from app.agents.security import SecurityAgent
from app.agents.send import SendAgent
from app.agents.timer import TimerAgent
from app.cache.cache_manager import CacheManager
from app.cache.embedding import get_embedding_engine
from app.cache.vector_store import COLLECTION_ENTITY_INDEX, get_vector_store
from app.db.repository import AgentConfigRepository, SettingsRepository, SetupStateRepository
from app.entity.aliases import AliasResolver
from app.entity.index import EntityIndex
from app.entity.ingest import parse_ha_states, state_to_entity_index_entry
from app.entity.matcher import EntityMatcher
from app.ha_client.home_context import home_context_provider
from app.ha_client.rest import HARestClient
from app.presence.detector import PresenceDetector

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


# P3-11: tunables for the runtime background loops. Kept module-level
# so they can be inspected / overridden from tests via monkeypatch
# without touching the call sites.
_ENTITY_SYNC_DEFAULT_INTERVAL_MIN = 30
_ENTITY_SYNC_DISABLED_RECHECK_SEC = 300
_ENTITY_UPDATE_FLUSH_INTERVAL_SEC = 0.5


def _set_entity_index_pending_status(entity_index: EntityIndex, *, state: str, total: int) -> None:
    """Mark entity index as building/syncing before background priming starts."""
    entity_index._status = {
        "state": state,
        "progress": 0,
        "total": total,
        "processed": 0,
        "error": None,
    }


async def _prime_entity_index(app: FastAPI, ha_client: HARestClient, entity_index: EntityIndex, vector_store) -> None:
    """Fetch HA states and build/sync the entity index in the background."""
    try:
        states = await ha_client.get_states()
        entities = parse_ha_states(states)
        existing_count = vector_store.count(COLLECTION_ENTITY_INDEX)
        if existing_count > 0:
            _set_entity_index_pending_status(entity_index, state="syncing", total=len(entities))
            result = await entity_index.sync_async(entities)
            logger.info(
                "Entity index synced in background (existing=%d): +%d ~%d -%d =%d",
                existing_count,
                result["added"],
                result["updated"],
                result["removed"],
                result["unchanged"],
            )
        else:
            _set_entity_index_pending_status(entity_index, state="building", total=len(entities))
            await entity_index.populate_async(entities)
            logger.info("Entity index populated in background with %d entities", len(entities))
    except Exception:
        logger.warning("Failed to prime entity index in background", exc_info=True)


async def schedule_entity_index_prime(
    app: FastAPI,
    ha_client: HARestClient,
    entity_index: EntityIndex,
    vector_store,
) -> bool:
    """Ensure a single background task exists to build/sync the entity index."""
    task = getattr(app.state, "entity_index_init_task", None)
    if task is not None and not task.done():
        return False
    app.state.entity_index_init_task = asyncio.create_task(
        _prime_entity_index(app, ha_client, entity_index, vector_store)
    )
    return True


async def _periodic_entity_sync(app: FastAPI) -> None:
    """Periodically sync entity index with Home Assistant state."""
    while True:
        try:
            raw = await SettingsRepository.get_value(
                "entity_sync.interval_minutes", str(_ENTITY_SYNC_DEFAULT_INTERVAL_MIN)
            )
            interval_minutes = int(raw)
        except (TypeError, ValueError):
            interval_minutes = _ENTITY_SYNC_DEFAULT_INTERVAL_MIN

        if interval_minutes <= 0:
            await asyncio.sleep(_ENTITY_SYNC_DISABLED_RECHECK_SEC)
            continue

        await asyncio.sleep(interval_minutes * 60)

        try:
            ha_client = app.state.ha_client
            entity_index = app.state.entity_index
            if not ha_client or not entity_index:
                continue

            states = await ha_client.get_states()
            entities = parse_ha_states(states)
            result = await entity_index.sync_async(entities)
            logger.info(
                "Periodic entity sync: +%d ~%d -%d =%d",
                result["added"],
                result["updated"],
                result["removed"],
                result["unchanged"],
            )
        except Exception:
            logger.warning("Periodic entity sync failed", exc_info=True)


async def _purge_stale_response_cache(cache_manager: CacheManager) -> None:
    """One-time startup task: purge stale read-only response cache entries."""
    try:
        count = await cache_manager.purge_readonly_entries()
        if count:
            logger.info("Purged %d stale read-only response cache entries", count)
        else:
            logger.info("No stale read-only response cache entries to purge")
    except Exception:
        logger.warning("Failed to purge stale response cache entries", exc_info=True)


def _create_phase2_agent(agent_id: str, app: FastAPI):
    """Instantiate a Phase 2 agent by ID for runtime registration."""
    agent_map = {
        "timer-agent": TimerAgent,
        "climate-agent": ClimateAgent,
        "media-agent": MediaAgent,
        "scene-agent": SceneAgent,
        "automation-agent": AutomationAgent,
        "security-agent": SecurityAgent,
        "send-agent": SendAgent,
    }
    with_matcher = {
        "climate-agent",
        "security-agent",
        "timer-agent",
        "scene-agent",
        "automation-agent",
        "media-agent",
    }

    cls = agent_map.get(agent_id)
    if cls is None:
        return None

    ha_client = getattr(app.state, "ha_client", None)
    entity_index = getattr(app.state, "entity_index", None)
    entity_matcher = getattr(app.state, "entity_matcher", None)

    if agent_id in with_matcher:
        return cls(ha_client=ha_client, entity_index=entity_index, entity_matcher=entity_matcher)
    return cls(ha_client=ha_client, entity_index=entity_index)


async def _initialize_setup_dependent_services(app: FastAPI, *, source: str) -> None:
    """Idempotent core of the setup-dependent bootstrap.

    FLOW-SETUP-1 (P1-2): single canonical implementation used by both the
    FastAPI ``lifespan`` on fresh container startup (``source="lifespan"``)
    and by :func:`ensure_setup_runtime_initialized` which runs after the
    user completes the setup wizard in a long-running process
    (``source="post-setup"``). Each step is individually idempotent --
    re-entering after a partial init reuses whatever ``app.state``
    already carries instead of re-instantiating.
    """
    registry = getattr(app.state, "registry", None)
    dispatcher = getattr(app.state, "dispatcher", None)
    mcp_registry = getattr(app.state, "mcp_registry", None)
    mcp_tool_manager = getattr(app.state, "mcp_tool_manager", None)
    if registry is None or dispatcher is None or mcp_registry is None or mcp_tool_manager is None:
        raise RuntimeError("Core runtime state is not ready for setup initialization")

    logger.info("Setup init (%s): initializing setup-dependent services", source)

    await get_embedding_engine()
    vector_store = await get_vector_store()

    ha_client = getattr(app.state, "ha_client", None)
    if ha_client is None:
        ha_client = HARestClient()
        await ha_client.initialize()
    else:
        await ha_client.reload()
    app.state.ha_client = ha_client

    entity_index = getattr(app.state, "entity_index", None)
    if entity_index is None:
        entity_index = EntityIndex(vector_store)
        app.state.entity_index = entity_index

    await schedule_entity_index_prime(app, ha_client, entity_index, vector_store)

    try:
        await home_context_provider.refresh(ha_client)
    except Exception:
        logger.warning("Setup init (%s): failed to pre-warm HomeContext cache", source, exc_info=True)

    alias_resolver = getattr(app.state, "alias_resolver", None)
    if alias_resolver is None:
        alias_resolver = AliasResolver()
        await alias_resolver.load()
        app.state.alias_resolver = alias_resolver

    entity_matcher = getattr(app.state, "entity_matcher", None)
    if entity_matcher is None:
        entity_matcher = EntityMatcher(entity_index, alias_resolver)
        await entity_matcher.load_config()
        app.state.entity_matcher = entity_matcher

    rewrite_agent = getattr(app.state, "rewrite_agent", None)
    if rewrite_agent is None:
        rewrite_agent = RewriteAgent(ha_client=ha_client, entity_index=entity_index)
        app.state.rewrite_agent = rewrite_agent

    cache_manager = getattr(app.state, "cache_manager", None)
    if cache_manager is None:
        cache_manager = CacheManager(vector_store, rewrite_agent=rewrite_agent)
        await cache_manager.initialize()
        app.state.cache_manager = cache_manager

    purge_task = getattr(app.state, "purge_task", None)
    if purge_task is None or purge_task.done():
        app.state.purge_task = asyncio.create_task(_purge_stale_response_cache(cache_manager))

    presence_detector = getattr(app.state, "presence_detector", None)
    if presence_detector is None:
        presence_detector = PresenceDetector(ha_client)
        await presence_detector.initialize()
        app.state.presence_detector = presence_detector

    try:
        await mcp_registry.load_from_db()
    except Exception:
        logger.warning("Setup init (%s): failed to load MCP servers from DB", source, exc_info=True)

    from app.db.repository import AgentMcpToolsRepository, McpServerRepository

    ddg_server = await McpServerRepository.get("duckduckgo-search")
    if ddg_server is None:
        logger.info("Setup init (%s): registering built-in DuckDuckGo MCP server", source)
        connected = await mcp_registry.add_server(
            name="duckduckgo-search",
            transport="stdio",
            command_or_url="python -m app.mcp.servers.duckduckgo_server",
        )
        if connected:
            try:
                client = mcp_registry.get_client("duckduckgo-search")
                if client:
                    tools = await client.list_tools()
                    for tool in tools:
                        await AgentMcpToolsRepository.assign_tool(
                            "general-agent",
                            "duckduckgo-search",
                            tool["name"],
                        )
                    logger.info("Assigned %d DuckDuckGo tools to general-agent", len(tools))
            except Exception:
                logger.warning(
                    "Setup init (%s): failed to auto-assign DuckDuckGo tools",
                    source,
                    exc_info=True,
                )
        else:
            logger.warning(
                "Setup init (%s): DuckDuckGo MCP server registered but failed to connect",
                source,
            )

    filler_agent = FillerAgent(ha_client=ha_client, entity_index=entity_index)
    orchestrator_agent = OrchestratorAgent(
        dispatcher=dispatcher,
        registry=registry,
        cache_manager=cache_manager,
        presence_detector=presence_detector,
        ha_client=ha_client,
        entity_index=entity_index,
        filler_agent=filler_agent,
    )
    await registry.register(orchestrator_agent)

    general_agent = GeneralAgent(
        ha_client=ha_client,
        entity_index=entity_index,
        mcp_tool_manager=mcp_tool_manager,
    )
    await registry.register(general_agent)

    light_agent = LightAgent(ha_client=ha_client, entity_index=entity_index, entity_matcher=entity_matcher)
    await registry.register(light_agent)

    music_agent = MusicAgent(ha_client=ha_client, entity_index=entity_index, entity_matcher=entity_matcher)
    await registry.register(music_agent)

    await registry.register(filler_agent)

    phase2_agents = [
        "timer-agent",
        "climate-agent",
        "media-agent",
        "scene-agent",
        "automation-agent",
        "security-agent",
        "send-agent",
    ]
    for agent_id in phase2_agents:
        config = await AgentConfigRepository.get(agent_id)
        if config and config.get("enabled"):
            agent = _create_phase2_agent(agent_id, app)
            if agent is not None:
                await registry.register(agent)

    await registry.register(rewrite_agent)

    custom_loader = getattr(app.state, "custom_loader", None)
    if custom_loader is None:
        custom_loader = CustomAgentLoader(registry, ha_client=ha_client, entity_index=entity_index)
        await custom_loader.load_all()
        app.state.custom_loader = custom_loader

    await orchestrator_agent.initialize()

    ws_client = getattr(app.state, "ws_client", None)
    if ws_client is None:
        from app.agents.timer_executor import _timer_pool, on_timer_finished
        from app.ha_client.websocket import HAWebSocketClient

        ws_client = HAWebSocketClient()
        entity_update_queue: asyncio.Queue = asyncio.Queue()

        async def _flush_entity_updates() -> None:
            while True:
                await asyncio.sleep(_ENTITY_UPDATE_FLUSH_INTERVAL_SEC)
                batch = []
                while not entity_update_queue.empty():
                    try:
                        batch.append(entity_update_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if batch:
                    try:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, entity_index.batch_add, batch)
                    except Exception:
                        logger.warning("Batch entity index update failed", exc_info=True)

        async def on_state_changed(event: dict) -> None:
            data = event.get("data", {})
            new_state = data.get("new_state")
            old_state = data.get("old_state")
            entity_id = data.get("entity_id", "")

            if new_state is None and old_state is not None:
                await entity_index.remove_async(entity_id)
            elif new_state is not None:
                entry = state_to_entity_index_entry(new_state, entity_id=entity_id)
                entity_update_queue.put_nowait(entry)

        def on_state_changed_presence(event: dict) -> None:
            data = event.get("data", {})
            entity_id = data.get("entity_id", "")
            new_state = data.get("new_state")
            if new_state and entity_id.startswith("binary_sensor."):
                area = new_state.get("attributes", {}).get("area_id")
                presence_detector.on_sensor_state_change(entity_id, new_state.get("state", "off"), area)

        async def _on_timer_finished_event(event: dict) -> None:
            data = event.get("data", {})
            entity_id = data.get("entity_id", "")
            if entity_id:
                await on_timer_finished(
                    entity_id,
                    ha_client,
                    entity_index=getattr(app.state, "entity_index", None),
                )

        async def _on_timer_cancelled_event(event: dict) -> None:
            data = event.get("data", {})
            entity_id = data.get("entity_id", "")
            if entity_id:
                _timer_pool.release(entity_id)

        ws_client.on_event("state_changed", on_state_changed)
        ws_client.on_event("state_changed", on_state_changed_presence)
        ws_client.on_event("timer.finished", _on_timer_finished_event)
        ws_client.on_event("timer.cancelled", _on_timer_cancelled_event)

        app.state.ws_client = ws_client
        app.state.ws_task = asyncio.create_task(ws_client.run())
        app.state.flush_task = asyncio.create_task(_flush_entity_updates())

    if ha_client is not None and ws_client is not None:
        ha_client.set_state_observer(ws_client)

    sync_task = getattr(app.state, "sync_task", None)
    if sync_task is None or sync_task.done():
        app.state.sync_task = asyncio.create_task(_periodic_entity_sync(app))

    alarm_monitor = getattr(app.state, "alarm_monitor", None)
    if alarm_monitor is None:
        from app.agents.alarm_monitor import AlarmMonitor

        alarm_monitor = AlarmMonitor(ha_client)
        await alarm_monitor.start()
        app.state.alarm_monitor = alarm_monitor

    logger.info("Setup init (%s): completed", source)


async def ensure_setup_runtime_initialized(app: FastAPI) -> bool:
    """Initialize setup-dependent runtime services after setup completion.

    Returns ``True`` when initialization work was performed in this call,
    otherwise ``False``.
    """
    if getattr(app.state, "setup_runtime_initialized", False):
        return False

    lock = getattr(app.state, "setup_runtime_init_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.setup_runtime_init_lock = lock

    async with lock:
        if getattr(app.state, "setup_runtime_initialized", False):
            return False
        if not await SetupStateRepository.is_complete():
            return False

        await _initialize_setup_dependent_services(app, source="post-setup")

        app.state.setup_runtime_initialized = True
        logger.info("Setup-dependent runtime initialized in-process")
        return True
