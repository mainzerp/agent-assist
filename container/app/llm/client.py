import asyncio
import json
import logging
from typing import Callable

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

        if not content:
            raise ValueError(
                f"Empty LLM response for agent={agent_id} after retry "
                f"(finish_reason={response.choices[0].finish_reason})"
            )
        return content
    except litellm.exceptions.AuthenticationError:
        logger.error("Authentication failed for agent=%s model=%s -- check API key",
                      agent_id, model)
        raise
    except Exception:
        logger.exception("LLM call failed for agent=%s model=%s", agent_id, model)
        raise


async def complete_with_tools(
    agent_id: str,
    messages: list[dict],
    tools: list[dict],
    tool_executor: Callable,
    max_tool_rounds: int = 5,
    **overrides: object,
) -> str:
    """LLM completion with tool/function calling loop.

    Parameters:
        agent_id: Agent ID for config lookup.
        messages: Conversation messages (system + user).
        tools: OpenAI-format tool schemas.
        tool_executor: Async callable (tool_name, arguments) -> str.
        max_tool_rounds: Max LLM<->tool round-trips (default 5).
        **overrides: Model/temperature/max_tokens overrides.

    Returns:
        Final text response from the LLM.
    """
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

    # Make a mutable copy of messages for the tool-call loop
    msgs = list(messages)

    for _round in range(max_tool_rounds):
        logger.debug(
            "LLM tool-call round %d: agent=%s model=%s",
            _round + 1, agent_id, model,
        )
        response = await litellm.acompletion(
            model=model,
            messages=msgs,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=temperature,
            **provider_params,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            # No tool calls -- return the text content
            content = msg.content
            if not content:
                logger.warning(
                    "Empty LLM response in tool-call loop for agent=%s round=%d",
                    agent_id, _round + 1,
                )
                return ""
            return content

        # Append the assistant message with tool_calls to the conversation
        msgs.append(msg)

        # Execute each tool call and append results
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, TypeError):
                fn_args = {}
                logger.warning("Failed to parse tool arguments for '%s'", fn_name)

            logger.debug("Executing tool '%s' with args: %s", fn_name, fn_args)

            try:
                result_str = await tool_executor(fn_name, fn_args)
            except Exception as e:
                logger.warning("Tool executor '%s' raised: %s", fn_name, e)
                result_str = f"Tool error: {e}"

            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Max rounds exhausted -- force a final text response without tools
    logger.warning(
        "Max tool rounds (%d) exhausted for agent=%s, forcing final response",
        max_tool_rounds, agent_id,
    )
    response = await litellm.acompletion(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
        **provider_params,
    )
    content = response.choices[0].message.content
    return content or ""
