"""Admin REST API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator

from app.security.auth import require_admin_session
from app.security.encryption import store_secret, retrieve_secret, delete_secret
from app.db.repository import (
    AgentConfigRepository,
    EntityMatchingConfigRepository,
    EntityVisibilityRepository,
    SecretsRepository,
    SettingsRepository,
)

logger = logging.getLogger(__name__)

# Maps provider name to its secret key in the secrets table
PROVIDER_SECRET_KEYS = {
    "openrouter": "openrouter_api_key",
    "groq": "groq_api_key",
    "anthropic": "anthropic_api_key",
}


class ProviderKeyUpdate(BaseModel):
    provider: str
    api_key: str


class OllamaUrlUpdate(BaseModel):
    url: str


class ProviderTestRequest(BaseModel):
    provider: str
    api_key: str | None = None


class SettingsUpdatePayload(BaseModel):
    """Validated settings update payload."""
    items: dict[str, Any]

    @field_validator("items")
    @classmethod
    def validate_items(cls, v):
        if not v:
            raise ValueError("items must not be empty")
        for key in v:
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError(f"Invalid setting key: {key}")
        return v

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin_session)])

# The registry is set by main.py during startup
_registry = None


def set_registry(reg) -> None:
    """Called by main.py to inject the A2A registry."""
    global _registry
    _registry = reg


@router.get("/agents")
async def list_agents():
    """List all agents (registered + disabled from DB)."""
    agents = await _registry.list_agents()
    seen_ids = set()
    result = []
    for a in agents:
        card = a.model_dump()
        config = await AgentConfigRepository.get(a.agent_id)
        if config:
            card.update(config)
        result.append(card)
        seen_ids.add(a.agent_id)

    # Known built-in agent IDs (from seed data)
    _BUILTIN_AGENTS = {
        "orchestrator", "general-agent", "light-agent", "music-agent",
        "timer-agent", "climate-agent", "media-agent", "scene-agent",
        "automation-agent", "security-agent", "rewrite-agent",
        "send-agent",
    }

    # Include disabled built-in agents from DB that are not yet registered
    all_configs = await AgentConfigRepository.list_all()
    for config in all_configs:
        aid = config["agent_id"]
        if aid not in seen_ids and aid in _BUILTIN_AGENTS:
            entry = {
                "agent_id": aid,
                "name": aid.replace("-", " ").title(),
                "description": config.get("description", ""),
                "skills": [],
                "input_types": ["text/plain"],
                "output_types": ["text/plain", "application/json"],
                "endpoint": f"local://{aid}",
            }
            entry.update(config)
            result.append(entry)
    return {"agents": result}


@router.get("/settings")
async def get_settings():
    """Get all settings grouped by category."""
    rows = await SettingsRepository.get_all()
    grouped: dict[str, list] = {}
    for row in rows:
        cat = row.get("category", "general")
        grouped.setdefault(cat, []).append(row)
    return {"settings": grouped}


def _validate_setting_value(key: str, value: str, value_type: str) -> None:
    """Validate a setting value against its stored type. Raises HTTPException on failure."""
    # COR-6: typed numeric/boolean settings must not accept the empty string,
    # otherwise the dashboard can blank out a value and silently store ""
    # which later coerces to a default in unrelated code paths.
    if value_type in ("int", "float", "bool") and value == "":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value for '{key}': empty string is not a valid {value_type}",
        )
    try:
        if value_type == "int":
            int(value)
        elif value_type == "float":
            float(value)
        elif value_type == "bool":
            if str(value).lower() not in ("true", "false", "1", "0"):
                raise ValueError("Expected boolean")
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value for '{key}': expected {value_type}",
        )


@router.put("/settings")
async def update_settings(payload: SettingsUpdatePayload):
    """Update multiple settings. Payload: {"items": {key: value, ...}}."""
    for key, value in payload.items.items():
        existing = await SettingsRepository.get(key)
        if existing is None:
            raise HTTPException(status_code=400, detail=f"Unknown setting key: {key}")
        # Validate value type against stored type
        value_type = existing.get("value_type", "str")
        _validate_setting_value(key, str(value), value_type)
        await SettingsRepository.set(key, str(value), value_type=existing["value_type"],
                                     category=existing.get("category", "general"),
                                     description=existing.get("description"))
    return {"status": "ok"}


@router.put("/settings/{key}")
async def update_single_setting(key: str, payload: dict):
    """Update a single setting by key."""
    value = payload.get("value")
    if value is None:
        return {"status": "error", "detail": "Missing value"}

    existing = await SettingsRepository.get(key)
    if existing is None:
        raise HTTPException(status_code=400, detail=f"Unknown setting key: {key}")

    value_type = existing.get("value_type", "str")
    _validate_setting_value(key, str(value), value_type)

    await SettingsRepository.set(key, str(value), value_type=existing["value_type"],
                                 category=existing.get("category", "general"),
                                 description=existing.get("description"))
    return {"status": "ok", "key": key}


@router.get("/entity-matching-weights")
async def get_entity_matching_weights():
    """Get all entity matching signal weights."""
    rows = await EntityMatchingConfigRepository.get_all()
    return {"weights": {row["key"]: row["value"] for row in rows}}


@router.put("/entity-matching-weights")
async def update_entity_matching_weights(payload: dict):
    """Update entity matching signal weights. Payload: {key: value, ...}."""
    allowed_keys = {
        "weight.levenshtein", "weight.jaro_winkler", "weight.phonetic",
        "weight.embedding", "weight.alias",
    }
    items = payload.get("items", payload)
    if isinstance(items, dict):
        for key, value in items.items():
            if key in allowed_keys:
                await EntityMatchingConfigRepository.set(key, str(value))
    return {"status": "ok"}


# =========================================================================
# LLM Provider Management
# =========================================================================


@router.get("/llm-providers")
async def get_llm_provider_status():
    """Return status of all LLM providers with masked keys."""
    stored_keys = await SecretsRepository.list_keys()
    providers: dict = {}
    for provider, secret_key in PROVIDER_SECRET_KEYS.items():
        configured = secret_key in stored_keys
        masked_key = None
        if configured:
            raw = await retrieve_secret(secret_key)
            if raw and len(raw) >= 4:
                masked_key = raw[-4:]
            elif raw:
                masked_key = "****"
        providers[provider] = {"configured": configured, "masked_key": masked_key}
    # Ollama
    ollama_url = await SettingsRepository.get_value("ollama_base_url")
    providers["ollama"] = {
        "configured": ollama_url is not None,
        "url": ollama_url,
    }
    return {"providers": providers}


@router.put("/llm-providers")
async def update_llm_provider_key(payload: ProviderKeyUpdate):
    """Save an encrypted API key for a provider."""
    if payload.provider not in PROVIDER_SECRET_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {payload.provider}")
    secret_key = PROVIDER_SECRET_KEYS[payload.provider]
    await store_secret(secret_key, payload.api_key)
    return {"status": "ok", "provider": payload.provider}


@router.put("/llm-providers/ollama")
async def update_ollama_url(payload: OllamaUrlUpdate):
    """Save the Ollama base URL."""
    await SettingsRepository.set("ollama_base_url", payload.url, "string", "llm", "Ollama API URL")
    return {"status": "ok"}


@router.delete("/llm-providers/{provider}")
async def delete_llm_provider_key(provider: str):
    """Remove a stored API key for a provider."""
    if provider not in PROVIDER_SECRET_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    await delete_secret(PROVIDER_SECRET_KEYS[provider])
    return {"status": "ok"}


@router.post("/llm-providers/test")
async def test_llm_provider(payload: ProviderTestRequest):
    """Test connectivity for an LLM provider."""
    import litellm

    provider = payload.provider
    api_key = payload.api_key

    test_models = {
        "groq": "groq/llama-3.1-8b-instant",
        "openrouter": "openrouter/openai/gpt-4o-mini",
        "anthropic": "anthropic/claude-3-haiku-20240307",
        "ollama": "ollama/llama3",
    }
    if provider not in test_models:
        return {"status": "error", "detail": f"Unknown provider: {provider}"}

    # If no key given, retrieve stored key
    if provider == "ollama":
        base_url = await SettingsRepository.get_value("ollama_base_url", "http://localhost:11434")
        api_key = "not-needed"
    elif not api_key:
        secret_key = PROVIDER_SECRET_KEYS.get(provider)
        if secret_key:
            api_key = await retrieve_secret(secret_key)
        if not api_key:
            return {"status": "error", "detail": "No API key configured for " + provider}

    try:
        kwargs: dict = {
            "model": test_models[provider],
            "messages": [{"role": "user", "content": "Say hello"}],
            "api_key": api_key,
            "max_tokens": 10,
        }
        if provider == "ollama":
            kwargs["api_base"] = base_url
        await litellm.acompletion(**kwargs)
        return {"status": "ok", "provider": provider}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/llm-providers/configured")
async def get_configured_providers():
    """Return all known providers with their configuration status."""
    stored_keys = await SecretsRepository.list_keys()
    configured = []
    all_providers = []
    for provider, secret_key in PROVIDER_SECRET_KEYS.items():
        all_providers.append(provider)
        if secret_key in stored_keys:
            configured.append(provider)
    all_providers.append("ollama")
    ollama_url = await SettingsRepository.get_value("ollama_base_url")
    if ollama_url:
        configured.append("ollama")
    return {"providers": all_providers, "configured": configured}


# =========================================================================
# Entity Visibility Summary
# =========================================================================


@router.get("/agents/visibility-summary")
async def get_all_agents_visibility_summary():
    """Return a summary of entity visibility domains per agent."""
    all_rules = await EntityVisibilityRepository.list_all()
    agent_rules: dict[str, list[dict]] = {}
    for rule in all_rules:
        agent_id = rule["agent_id"]
        agent_rules.setdefault(agent_id, []).append(rule)

    summary: dict[str, dict] = {}
    for agent_id, rules in agent_rules.items():
        domains: set[str] = set()
        excluded_domains: set[str] = set()
        device_classes: set[str] = set()
        excluded_device_classes: set[str] = set()
        for r in rules:
            if r["rule_type"] == "domain_include":
                domains.add(r["rule_value"])
            elif r["rule_type"] == "domain_exclude":
                excluded_domains.add(r["rule_value"])
            elif r["rule_type"] == "area_include":
                domains.add("area:" + r["rule_value"])
            elif r["rule_type"] == "area_exclude":
                excluded_domains.add("area:" + r["rule_value"])
            elif r["rule_type"] == "entity_include":
                domain_part = r["rule_value"].split(".")[0] if "." in r["rule_value"] else r["rule_value"]
                domains.add(domain_part)
            elif r["rule_type"] == "entity_exclude":
                domain_part = r["rule_value"].split(".")[0] if "." in r["rule_value"] else r["rule_value"]
                excluded_domains.add(domain_part)
            elif r["rule_type"] == "device_class_include":
                device_classes.add(r["rule_value"])
            elif r["rule_type"] == "device_class_exclude":
                excluded_device_classes.add(r["rule_value"])
        summary[agent_id] = {
            "domains": sorted(domains),
            "excluded_domains": sorted(excluded_domains),
            "device_classes": sorted(device_classes),
            "excluded_device_classes": sorted(excluded_device_classes),
            "has_rules": True,
        }
    return {"summary": summary}


@router.get("/timers")
async def get_timers_info(request: Request):
    """Return timer, alarm, pool, and delayed task state for the dashboard."""
    from app.agents.timer_executor import _timer_pool
    from app.agents.delayed_tasks import delayed_task_manager

    ha_client = getattr(request.app.state, "ha_client", None)

    timers = []
    alarms = []

    if ha_client:
        try:
            states = await ha_client.get_states()
        except Exception:
            states = []

        for s in states:
            entity_id = s.get("entity_id", "")
            state = s.get("state", "unknown")
            attrs = s.get("attributes", {})
            friendly_name = attrs.get("friendly_name", entity_id)

            if entity_id.startswith("timer."):
                pool_name = _timer_pool.get_name(entity_id)
                duration = attrs.get("duration", "")
                remaining = attrs.get("remaining", "")
                timers.append({
                    "entity_id": entity_id,
                    "name": pool_name or friendly_name,
                    "friendly_name": friendly_name,
                    "pool_name": pool_name,
                    "state": state,
                    "duration": duration,
                    "remaining": remaining,
                })

            elif entity_id.startswith("input_datetime."):
                has_date = attrs.get("has_date", False)
                has_time = attrs.get("has_time", False)
                dtype = "datetime" if (has_date and has_time) else ("date" if has_date else "time")
                alarms.append({
                    "entity_id": entity_id,
                    "name": friendly_name,
                    "state": state,
                    "type": dtype,
                })

    # Timer pool
    pool_mappings = _timer_pool.all_mappings()
    pool = {
        "mappings": [
            {"name": name, "entity_id": eid}
            for name, eid in pool_mappings.items()
        ],
        "allocated": len(pool_mappings),
    }

    # Delayed tasks
    pending = delayed_task_manager.get_pending()

    return {
        "timers": timers,
        "alarms": alarms,
        "pool": pool,
        "delayed_tasks": pending,
    }


@router.get("/fernet-key-backup")
async def get_fernet_key_backup():
    """Export the Fernet key for backup. Handle with extreme care."""
    from app.security.encryption import export_fernet_key
    return {"key": export_fernet_key(), "warning": "Store this key securely. Loss of this key makes all encrypted secrets unrecoverable."}


# =========================================================================
# Notification Profile
# =========================================================================


@router.get("/notification-profile")
async def get_notification_profile():
    """Get current notification profile."""
    import json as _json
    raw = await SettingsRepository.get_value("notification.profile")
    if raw:
        return {"profile": _json.loads(raw)}
    return {"profile": {}}


@router.put("/notification-profile")
async def update_notification_profile(payload: dict):
    """Update notification profile."""
    import json as _json
    profile = payload.get("profile", payload)
    await SettingsRepository.set(
        "notification.profile",
        _json.dumps(profile),
        value_type="json",
        category="notification",
        description="Timer/alarm notification profile: channels and targets",
    )
    return {"status": "ok"}


# =========================================================================
# Alarm Monitor
# =========================================================================


@router.get("/alarm-monitor")
async def get_alarm_monitor_status(request: Request):
    """Get alarm monitor status."""
    alarm_monitor = getattr(request.app.state, "alarm_monitor", None)
    if not alarm_monitor:
        return {"active": False, "fired_today": [], "check_interval": 30}
    return {
        "active": True,
        "fired_today": alarm_monitor.fired_today,
        "check_interval": 30,
    }


# =========================================================================
# Recently Expired Timers
# =========================================================================


@router.get("/timers/recently-expired")
async def get_recently_expired_timers():
    """Get recently expired timers."""
    from app.agents.timer_executor import get_recently_expired
    expired = get_recently_expired()
    return {
        "recently_expired": [
            {
                "name": e.name,
                "entity_id": e.entity_id,
                "expired_at": e.expired_at.isoformat(),
            }
            for e in expired
        ],
    }
