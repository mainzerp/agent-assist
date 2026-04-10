import asyncio
import logging

import litellm

from app.db.repository import AgentConfigRepository
from app.llm.providers import resolve_provider_params
from app.models.agent import AgentConfig

logger = logging.getLogger(__name__)

# Suppress litellm's internal verbose logging unless user wants debug.
litellm.suppress_debug_info = True


async def complete(
    agent_id: str,
    messages: list[dict],
    **overrides: object,
) -> str:
    row = await AgentConfigRepository.get(agent_id)
    if row is None:
        raise ValueError(f"No config found for agent: {agent_id}")
    config = AgentConfig(**row)

    model = overrides.get("model") or config.model
    if model is None:
        raise ValueError(f"No model configured for agent: {agent_id}")
    max_tokens = overrides.get("max_tokens", config.max_tokens)
    temperature = overrides.get("temperature", config.temperature)

    provider_params = await resolve_provider_params(model)

    logger.debug("LLM call: agent=%s model=%s tokens=%s temp=%s",
                 agent_id, model, max_tokens, temperature)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **provider_params,
        )
        content = response.choices[0].message.content

        # Single retry on empty response (e.g. rate limiting)
        if not content:
            logger.warning(
                "Empty LLM response for agent=%s model=%s finish_reason=%s, retrying once after 1s",
                agent_id, model, response.choices[0].finish_reason,
            )
            await asyncio.sleep(1)
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **provider_params,
            )
            content = response.choices[0].message.content

        return content
    except litellm.exceptions.AuthenticationError:
        logger.error("Authentication failed for agent=%s model=%s -- check API key",
                      agent_id, model)
        raise
    except Exception:
        logger.exception("LLM call failed for agent=%s model=%s", agent_id, model)
        raise
