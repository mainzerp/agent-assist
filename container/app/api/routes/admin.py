"""Admin REST API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

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

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin_session)])

# The registry is set by main.py during startup
_registry = None


def set_registry(reg) -> None:
    """Called by main.py to inject the A2A registry."""
    global _registry
    _registry = reg


@router.get("/agents")
async def list_agents():
    """List all registered agents."""
    agents = await _registry.list_agents()
    result = []
    for a in agents:
        card = a.model_dump()
        config = await AgentConfigRepository.get(a.agent_id)
        if config:
            card.update(config)
        result.append(card)
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


@router.put("/settings")
async def update_settings(payload: dict):
    """Update multiple settings. Payload: {key: value, ...}."""
    items = payload.get("items", payload)
    if isinstance(items, dict):
        for key, value in items.items():
            if key == "items":
                continue
            await SettingsRepository.set(key, str(value))
    return {"status": "ok"}


@router.put("/settings/{key}")
async def update_single_setting(key: str, payload: dict):
    """Update a single setting by key."""
    value = payload.get("value")
    if value is None:
        return {"status": "error", "detail": "Missing value"}
    value_type = payload.get("value_type", "string")
    category = payload.get("category", "general")
    description = payload.get("description")
    await SettingsRepository.set(key, str(value), value_type, category, description)
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
        for r in rules:
            if r["rule_type"] == "domain":
                domains.add(r["rule_value"])
            elif r["rule_type"] == "entity":
                parts = r["rule_value"].split(".")
                if parts:
                    domains.add(parts[0])
            elif r["rule_type"] == "area":
                domains.add("area:" + r["rule_value"])
        summary[agent_id] = {"domains": sorted(domains), "has_rules": True}
    return {"summary": summary}
