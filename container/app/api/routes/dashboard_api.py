"""Admin dashboard API endpoints.

Provides data for the HTMX-powered dashboard pages: overview metrics,
agent CRUD, prompt editing, extended health, and rewrite configuration.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.security.auth import require_admin_session
from app.a2a.protocol import JsonRpcRequest
from app.models.conversation import StreamToken
from app.models.agent import AgentTask
from app.db.repository import (
    AgentConfigRepository,
    SettingsRepository,
    AnalyticsRepository,
    TraceSummaryRepository,
    ConversationRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin",
    tags=["admin-dashboard"],
    dependencies=[Depends(require_admin_session)],
)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_SAFE_AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# --- Chat dispatcher (injected by main.py) ---

_dispatcher = None


def set_chat_dispatcher(dispatcher) -> None:
    """Called by main.py to inject the A2A dispatcher for chat bridge."""
    global _dispatcher
    _dispatcher = dispatcher


def _validate_agent_path(agent_id: str) -> Path:
    """Validate agent_id and return a safe prompt path within PROMPTS_DIR."""
    if not _SAFE_AGENT_ID_RE.match(agent_id):
        raise ValueError("Invalid agent ID")
    filename = agent_id.replace("-agent", "") + ".txt"
    prompt_path = (PROMPTS_DIR / filename).resolve()
    if not str(prompt_path).startswith(str(PROMPTS_DIR.resolve())):
        raise ValueError("Invalid agent ID")
    return prompt_path


# --- Request models ---

class AgentConfigUpdate(BaseModel):
    enabled: bool | None = None
    model: str | None = None
    timeout: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None
    description: str | None = None


class PromptUpdate(BaseModel):
    content: str


class RewriteConfigUpdate(BaseModel):
    model: str | None = None
    temperature: float | None = None


class PersonalityConfigUpdate(BaseModel):
    prompt: str | None = None


# --- Overview ---

@router.get("/overview")
async def get_overview(request: Request):
    """Aggregated overview metrics for the dashboard home page."""
    registry = request.app.state.registry
    entity_index = request.app.state.entity_index
    cache_manager = request.app.state.cache_manager
    mcp_registry = request.app.state.mcp_registry
    presence_detector = request.app.state.presence_detector

    agents = await registry.list_agents() if registry else []

    cache_stats = {}
    if cache_manager:
        try:
            cache_stats = cache_manager.get_stats()
        except Exception:
            pass

    entity_count = 0
    if entity_index:
        try:
            stats = entity_index.get_stats()
            entity_count = stats.get("count", 0)
        except Exception:
            pass

    mcp_count = 0
    if mcp_registry:
        try:
            servers = mcp_registry.list_servers()
            mcp_count = len(servers)
        except Exception:
            pass

    presence_rooms = 0
    if presence_detector:
        try:
            room_conf = presence_detector.get_room_confidence()
            presence_rooms = len(room_conf)
        except Exception:
            pass

    # Count recent requests from analytics (last 24h)
    recent_requests = 0
    try:
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        events = await AnalyticsRepository.query_by_range(
            event_type="request", start=start, limit=100000,
        )
        recent_requests = len(events)
    except Exception:
        pass

    # Compute cache hit rate from analytics DB (cache tier stats don't track hits/queries)
    cache_hit_rate = 0
    try:
        from datetime import datetime, timedelta, timezone as tz
        start_cache = (datetime.now(tz.utc) - timedelta(hours=24)).isoformat()
        cache_events = await AnalyticsRepository.query_by_range(
            event_type="cache", start=start_cache, limit=100000,
        )
        if cache_events:
            total_lookups = len(cache_events)
            hits = sum(
                1 for e in cache_events
                if e.get("hit_type", "") in ("routing_hit", "response_hit", "response_partial")
            )
            cache_hit_rate = round(hits / total_lookups * 100, 1)
    except Exception:
        pass
    hit_rate = cache_hit_rate

    return {
        "recent_requests": recent_requests,
        "cache_hit_rate": hit_rate,
        "agent_count": len(agents),
        "entity_count": entity_count,
        "mcp_server_count": mcp_count,
        "presence_rooms": presence_rooms,
    }


@router.get("/overview/extended")
async def get_overview_extended(request: Request):
    """Aggregated overview data for the redesigned dashboard home page.

    Returns all data the overview needs in a single call: metrics, agent
    distribution, cache tier stats, recent traces, and error/warning info.
    """
    registry = request.app.state.registry
    entity_index = request.app.state.entity_index
    mcp_registry = request.app.state.mcp_registry
    presence_detector = request.app.state.presence_detector

    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    start_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    # --- Basic counts (reuse existing overview logic) ---
    agents = await registry.list_agents() if registry else []

    entity_count = 0
    if entity_index:
        try:
            stats = entity_index.get_stats()
            entity_count = stats.get("count", 0)
        except Exception:
            pass

    mcp_count = 0
    if mcp_registry:
        try:
            servers = mcp_registry.list_servers()
            mcp_count = len(servers)
        except Exception:
            pass

    presence_rooms = 0
    if presence_detector:
        try:
            room_conf = presence_detector.get_room_confidence()
            presence_rooms = len(room_conf)
        except Exception:
            pass

    # --- Analytics: requests, latency ---
    requests = []
    try:
        requests = await AnalyticsRepository.query_by_range(
            event_type="request", start=start_24h, limit=100000,
        )
    except Exception:
        pass

    recent_requests = len(requests)
    latencies = [
        r["data"]["latency_ms"] for r in requests
        if r.get("data") and isinstance(r["data"], dict) and "latency_ms" in r["data"]
    ]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0

    # --- Cache events (all non-request events for cache analysis) ---
    all_events = []
    try:
        all_events = await AnalyticsRepository.query_by_range(
            start=start_24h, limit=100000,
        )
    except Exception:
        pass

    hit_types = {"routing_hit", "response_hit", "response_partial"}
    miss_types = {"miss"}
    hits = sum(1 for e in all_events if e.get("event_type") in hit_types)
    misses = sum(1 for e in all_events if e.get("event_type") in miss_types)
    total_cache = hits + misses
    cache_hit_rate = round(hits / total_cache * 100, 1) if total_cache > 0 else 0

    # Cache tier breakdown counts
    routing_hits = sum(1 for e in all_events if e.get("event_type") == "routing_hit")
    response_hits = sum(
        1 for e in all_events
        if e.get("event_type") in ("response_hit", "response_partial")
    )
    cache_misses = misses

    # Conversations count
    total_conversations = 0
    try:
        total_conversations = await ConversationRepository.count()
    except Exception:
        pass

    # --- Agent distribution ---
    agent_counts: dict[str, int] = defaultdict(int)
    agent_latencies_map: dict[str, list] = defaultdict(list)
    for e in requests:
        agent = e.get("agent_id") or "unknown"
        agent_counts[agent] += 1
        data = e.get("data")
        if isinstance(data, dict) and "latency_ms" in data:
            agent_latencies_map[agent].append(data["latency_ms"])

    agent_distribution = []
    for agent_id in sorted(agent_counts.keys()):
        lats = agent_latencies_map.get(agent_id, [])
        agent_distribution.append({
            "agent_id": agent_id,
            "request_count": agent_counts[agent_id],
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
        })

    # --- Request time-series (hourly buckets, last 24h) ---
    request_buckets: dict[str, int] = defaultdict(int)
    bucket_minutes = 60
    for e in requests:
        ts = e.get("created_at", "")
        try:
            dt = datetime.fromisoformat(ts)
            bucket_secs = bucket_minutes * 60
            ts_epoch = int(dt.timestamp())
            bucket_start = ts_epoch - (ts_epoch % bucket_secs)
            bucket_label = datetime.fromtimestamp(
                bucket_start, tz=timezone.utc
            ).strftime("%H:%M")
            request_buckets[bucket_label] += 1
        except (ValueError, TypeError):
            pass

    request_labels = sorted(request_buckets.keys())
    request_data = [request_buckets[lb] for lb in request_labels]
    request_trend = {"labels": request_labels, "data": request_data}

    # --- Recent traces (last 8) ---
    recent_traces = []
    try:
        result = await TraceSummaryRepository.list_filtered(
            page=1, per_page=8,
        )
        for t in result:
            recent_traces.append({
                "trace_id": t.get("trace_id", ""),
                "created_at": t.get("created_at", ""),
                "user_input": (t.get("user_input") or "")[:120],
                "routing_agent": t.get("routing_agent", ""),
                "total_duration_ms": t.get("total_duration_ms", 0),
                "label": t.get("label", ""),
            })
    except Exception:
        pass

    # --- Errors/warnings ---
    agent_timeouts = sum(
        1 for e in all_events if e.get("event_type") == "agent_timeout"
    )
    rewrite_events = [
        e for e in all_events if e.get("event_type") == "rewrite_invocation"
    ]
    rewrite_failures = sum(
        1 for e in rewrite_events
        if e.get("data") and isinstance(e["data"], dict)
        and not e["data"].get("success", True)
    )

    return {
        "recent_requests": recent_requests,
        "cache_hit_rate": cache_hit_rate,
        "agent_count": len(agents),
        "entity_count": entity_count,
        "mcp_server_count": mcp_count,
        "presence_rooms": presence_rooms,
        "avg_latency_ms": avg_latency,
        "total_conversations": total_conversations,
        "agent_distribution": agent_distribution,
        "cache_tier": {
            "routing_hits": routing_hits,
            "response_hits": response_hits,
            "misses": cache_misses,
        },
        "request_trend": request_trend,
        "recent_traces": recent_traces,
        "warnings": {
            "agent_timeouts": agent_timeouts,
            "rewrite_failures": rewrite_failures,
        },
    }


# --- Agent CRUD ---

@router.get("/agents/{agent_id}")
async def get_agent_config(agent_id: str):
    """Get a single agent configuration."""
    config = await AgentConfigRepository.get(agent_id)
    if config is None:
        return JSONResponse(status_code=404, content={"detail": "Agent not found"})
    return config


@router.put("/agents/{agent_id}")
async def update_agent_config(agent_id: str, payload: AgentConfigUpdate):
    """Update agent configuration fields."""
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        return {"status": "no changes"}
    await AgentConfigRepository.upsert(agent_id, **updates)
    return {"status": "ok", "agent_id": agent_id}


# --- Prompt read/write ---

@router.get("/agents/{agent_id}/prompt")
async def get_agent_prompt(agent_id: str):
    """Read the prompt file for an agent."""
    try:
        prompt_path = _validate_agent_path(agent_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Invalid agent ID"})
    if not prompt_path.is_file():
        return JSONResponse(status_code=404, content={"detail": "Prompt file not found"})
    content = await asyncio.to_thread(prompt_path.read_text, encoding="utf-8")
    filename = prompt_path.name
    return {"agent_id": agent_id, "filename": filename, "content": content}


@router.put("/agents/{agent_id}/prompt")
async def update_agent_prompt(agent_id: str, payload: PromptUpdate):
    """Write prompt file content for an agent."""
    try:
        prompt_path = _validate_agent_path(agent_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Invalid agent ID"})
    await asyncio.to_thread(prompt_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(prompt_path.write_text, payload.content, encoding="utf-8")
    filename = prompt_path.name
    return {"status": "ok", "agent_id": agent_id, "filename": filename}


# --- Extended health ---

@router.get("/health/extended")
async def get_extended_health(request: Request):
    """Extended health check for all subsystems."""
    ha_client = request.app.state.ha_client
    entity_index = request.app.state.entity_index
    cache_manager = request.app.state.cache_manager
    mcp_registry = request.app.state.mcp_registry
    startup_time = getattr(request.app.state, "startup_time", None)

    components = {}

    # HA connection
    try:
        if ha_client:
            await ha_client.get_states()
            components["ha_connection"] = {"status": "healthy"}
        else:
            components["ha_connection"] = {"status": "error", "detail": "Not initialized"}
    except Exception as exc:
        components["ha_connection"] = {"status": "error", "detail": str(exc)}

    # Entity index
    try:
        if entity_index:
            stats = entity_index.get_stats()
            components["entity_index"] = {"status": "healthy", "count": stats.get("count", 0)}
        else:
            components["entity_index"] = {"status": "error", "detail": "Not initialized"}
    except Exception as exc:
        components["entity_index"] = {"status": "error", "detail": str(exc)}

    # Cache
    try:
        if cache_manager:
            stats = cache_manager.get_stats()
            components["cache"] = {"status": "healthy", "stats": stats}
        else:
            components["cache"] = {"status": "error", "detail": "Not initialized"}
    except Exception as exc:
        components["cache"] = {"status": "error", "detail": str(exc)}

    # MCP servers
    try:
        if mcp_registry:
            servers = mcp_registry.list_servers()
            components["mcp_servers"] = {"status": "healthy", "count": len(servers)}
        else:
            components["mcp_servers"] = {"status": "error", "detail": "Not initialized"}
    except Exception as exc:
        components["mcp_servers"] = {"status": "error", "detail": str(exc)}

    # Uptime
    if startup_time:
        uptime_s = int(time.time() - startup_time)
        hours, remainder = divmod(uptime_s, 3600)
        minutes, seconds = divmod(remainder, 60)
        components["uptime"] = {"status": "healthy", "seconds": uptime_s, "display": f"{hours}h {minutes}m {seconds}s"}
    else:
        components["uptime"] = {"status": "unknown"}

    return components


# --- Rewrite config ---

@router.get("/rewrite/config")
async def get_rewrite_config():
    """Get current rewrite agent settings."""
    model = await SettingsRepository.get_value("rewrite.model", "")
    temperature = await SettingsRepository.get_value("rewrite.temperature", "0.7")
    return {
        "model": model or "",
        "temperature": float(temperature),
    }


@router.put("/rewrite/config")
async def update_rewrite_config(payload: RewriteConfigUpdate):
    """Update rewrite agent settings."""
    if payload.model is not None:
        await SettingsRepository.set(
            "rewrite.model", payload.model,
            "string", "rewrite", "Rewrite LLM model",
        )
    if payload.temperature is not None:
        await SettingsRepository.set(
            "rewrite.temperature", str(payload.temperature),
            "float", "rewrite", "Rewrite temperature",
        )
    return {"status": "ok"}


# --- Personality config ---

@router.get("/personality/config")
async def get_personality_config():
    """Get current personality prompt."""
    prompt = await SettingsRepository.get_value("personality.prompt", "")
    return {"prompt": prompt}


@router.put("/personality/config")
async def update_personality_config(payload: PersonalityConfigUpdate):
    """Save personality prompt."""
    if payload.prompt is not None:
        await SettingsRepository.set(
            "personality.prompt", payload.prompt,
            "string", "personality", "Personality system prompt for response mediation",
        )
    return {"status": "ok"}


# --- Chat bridge ---

class ChatRequest(BaseModel):
    text: str
    conversation_id: str | None = None
    language: str = "en"


@router.post("/chat")
async def admin_chat(request: Request, payload: ChatRequest):
    """Bridge: session-auth chat -> internal conversation pipeline."""
    if _dispatcher is None:
        return JSONResponse(status_code=503, content={"detail": "Dispatcher not ready"})

    span_collector = getattr(request.state, "span_collector", None)
    if span_collector:
        span_collector.source = "chat"
    task = AgentTask(
        description=payload.text,
        user_text=payload.text,
        conversation_id=payload.conversation_id,
    )
    a2a_request = JsonRpcRequest(
        method="message/send",
        params={
            "agent_id": "orchestrator",
            "task": task.model_dump(),
            "_span_collector": span_collector,
        },
        id=str(uuid.uuid4()),
    )
    response = await _dispatcher.dispatch(a2a_request)

    if response.error:
        return {"speech": f"Error: {response.error.message}", "conversation_id": payload.conversation_id}

    result = response.result or {}
    return {
        "speech": result.get("speech", ""),
        "conversation_id": payload.conversation_id,
    }


@router.post("/chat/stream")
async def admin_chat_stream(request: Request, payload: ChatRequest):
    """Bridge: session-auth SSE chat -> internal conversation pipeline (streaming)."""
    if _dispatcher is None:
        return JSONResponse(status_code=503, content={"detail": "Dispatcher not ready"})

    span_collector = getattr(request.state, "span_collector", None)
    if span_collector:
        span_collector.source = "chat"
    task = AgentTask(
        description=payload.text,
        user_text=payload.text,
        conversation_id=payload.conversation_id,
    )
    a2a_request = JsonRpcRequest(
        method="message/stream",
        params={
            "agent_id": "orchestrator",
            "task": task.model_dump(),
            "_span_collector": span_collector,
        },
        id=str(uuid.uuid4()),
    )

    async def generate():
        root_span_id = getattr(request.state, "root_span_id", None)
        if span_collector and root_span_id:
            span_collector._span_stack.append(root_span_id)
        try:
            async for chunk in _dispatcher.dispatch_stream(a2a_request):
                token = StreamToken(
                    token=chunk.result.get("token", ""),
                    done=chunk.done,
                    conversation_id=payload.conversation_id if chunk.done else None,
                )
                yield f"data: {token.model_dump_json()}\n\n"
        finally:
            if span_collector and root_span_id and root_span_id in span_collector._span_stack:
                span_collector._span_stack.remove(root_span_id)
            if span_collector:
                await span_collector.flush()

    return StreamingResponse(generate(), media_type="text/event-stream")
