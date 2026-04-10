"""Orchestrator agent for intent classification and task dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import AsyncGenerator

from app.agents.base import BaseAgent
from app.a2a.protocol import JsonRpcRequest
from app.db.repository import SettingsRepository
from app.analytics.collector import track_request, track_agent_timeout
from app.models.agent import AgentCard, AgentTask, TaskContext

logger = logging.getLogger(__name__)

# Max conversation turns to keep per conversation
_MAX_TURNS = 3

# Fallback agent when classification fails
_FALLBACK_AGENT = "general-agent"


class OrchestratorAgent(BaseAgent):
    """Classifies user intent and dispatches to specialized agents via A2A."""

    def __init__(self, dispatcher, registry=None, cache_manager=None, ha_client=None, entity_index=None, presence_detector=None) -> None:
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._dispatcher = dispatcher
        self._registry = registry
        self._cache_manager = cache_manager
        self._presence_detector = presence_detector
        self._conversations: dict[str, list[dict]] = defaultdict(list)
        self._default_timeout: int = 5
        self._max_iterations: int = 3

    async def initialize(self) -> None:
        """Load reliability config from DB. Call during startup."""
        await self._load_reliability_config()

    async def _load_reliability_config(self) -> None:
        """Read timeout and max_iterations from settings."""
        try:
            val = await SettingsRepository.get_value("a2a.default_timeout", "5")
            self._default_timeout = int(val)
        except (ValueError, TypeError):
            pass
        try:
            val = await SettingsRepository.get_value("a2a.max_iterations", "3")
            self._max_iterations = int(val)
        except (ValueError, TypeError):
            pass
        logger.info(
            "Orchestrator reliability config: timeout=%ds max_iterations=%d",
            self._default_timeout, self._max_iterations,
        )

    async def _get_known_agents(self) -> set[str]:
        """Return set of currently registered agent IDs (excluding orchestrator)."""
        if not self._registry:
            return {"light-agent", "music-agent", "general-agent"}
        cards = await self._registry.list_agents()
        return {card.agent_id for card in cards if card.agent_id != "orchestrator"}

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="orchestrator",
            name="Orchestrator",
            description="Routes user requests to the appropriate specialized agent.",
            skills=["intent_classification", "task_routing"],
            endpoint="local://orchestrator",
        )

    async def handle_task(self, task: AgentTask) -> dict:
        user_text = task.user_text or task.description
        conversation_id = task.conversation_id

        # Get span collector from task context if available
        span_collector = getattr(task, "_span_collector", None)

        # 1. Classify intent and get condensed task
        if span_collector:
            async with span_collector.start_span("classify", agent_id="orchestrator") as span:
                target_agent, condensed_task = await self._classify(user_text)
                span["metadata"]["target_agent"] = target_agent
        else:
            target_agent, condensed_task = await self._classify(user_text)
        logger.info(
            "Routed to %s: %s (conversation=%s)",
            target_agent, condensed_task[:80], conversation_id,
        )

        # 2. Build context with conversation turns
        turns = self._get_turns(conversation_id)
        context = TaskContext(conversation_turns=turns)

        # Inject presence room if available
        if self._presence_detector:
            room = self._presence_detector.get_most_likely_room()
            if room:
                context.presence_room = room

        # 3. Build task for target agent
        # description = condensed task (PRIMARY input for the agent)
        # user_text = original unmodified user text (FALLBACK ONLY)
        agent_task = AgentTask(
            description=condensed_task,
            user_text=user_text,
            conversation_id=conversation_id,
            context=context,
        )

        # 4. Dispatch via A2A message/send with timeout
        request = JsonRpcRequest(
            method="message/send",
            params={"agent_id": target_agent, "task": agent_task.model_dump()},
            id=conversation_id or "orchestrator-dispatch",
        )
        try:
            t0 = time.perf_counter()
            if span_collector:
                async with span_collector.start_span("dispatch", agent_id=target_agent) as span:
                    response = await asyncio.wait_for(
                        self._dispatcher.dispatch(request),
                        timeout=self._default_timeout,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    span["metadata"]["latency_ms"] = round(latency_ms, 1)
            else:
                response = await asyncio.wait_for(
                    self._dispatcher.dispatch(request),
                    timeout=self._default_timeout,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("Agent %s responded in %.1fms", target_agent, latency_ms)
            await track_request(target_agent, cache_hit=False, latency_ms=latency_ms)
        except asyncio.TimeoutError:
            logger.warning(
                "Agent %s timed out after %ds, falling back",
                target_agent, self._default_timeout,
            )
            await track_agent_timeout(target_agent, self._default_timeout)
            if target_agent != _FALLBACK_AGENT:
                request.params["agent_id"] = _FALLBACK_AGENT
                response = await self._dispatcher.dispatch(request)
            else:
                return {
                    "speech": "I couldn't process that request in time.",
                    "routed_to": target_agent,
                    "action_executed": None,
                }

        # 5. Extract result
        if response.error:
            logger.warning(
                "Agent %s error: %s -- falling back to %s",
                target_agent, response.error.message, _FALLBACK_AGENT,
            )
            if target_agent != _FALLBACK_AGENT:
                request.params["agent_id"] = _FALLBACK_AGENT
                response = await self._dispatcher.dispatch(request)

        result = response.result or {}
        speech = result.get("speech", "")

        # 6. Store conversation turn
        self._store_turn(conversation_id, user_text, speech)

        return {
            "speech": speech,
            "routed_to": target_agent,
            "action_executed": result.get("action_executed"),
        }

    async def handle_task_stream(self, task: AgentTask) -> AsyncGenerator[dict, None]:
        user_text = task.user_text or task.description
        conversation_id = task.conversation_id

        # 1. Classify (non-streaming -- fast via Groq)
        target_agent, condensed_task = await self._classify(user_text)
        logger.info(
            "Stream routed to %s: %s (conversation=%s)",
            target_agent, condensed_task[:80], conversation_id,
        )

        # 2. Build context and task
        turns = self._get_turns(conversation_id)
        context = TaskContext(conversation_turns=turns)
        agent_task = AgentTask(
            description=condensed_task,
            user_text=user_text,
            conversation_id=conversation_id,
            context=context,
        )

        # 3. Dispatch via A2A message/stream
        request = JsonRpcRequest(
            method="message/stream",
            params={"agent_id": target_agent, "task": agent_task.model_dump()},
            id=conversation_id or "orchestrator-stream",
        )

        collected_speech = []
        async for chunk in self._dispatcher.dispatch_stream(request):
            token = chunk.result.get("token", "")
            done = chunk.result.get("done", False)
            if token:
                collected_speech.append(token)
            yield {
                "token": token,
                "done": done,
                "conversation_id": conversation_id if done else None,
            }

        # 4. Store conversation turn
        full_speech = "".join(collected_speech)
        self._store_turn(conversation_id, user_text, full_speech)

    async def _classify(self, user_text: str) -> tuple[str, str]:
        """Classify user intent and produce a condensed task.

        The condensed task is a clear, actionable English description of
        what the agent should do. All entity/device/room/location names
        from the user's original text are preserved EXACTLY (verbatim,
        never translated or normalized).

        Returns:
            (target_agent_id, condensed_task) tuple.
        """
        # Check routing cache before calling LLM
        if self._cache_manager:
            try:
                cache_result = await self._cache_manager.process(user_text)
                if cache_result.hit_type == "routing_hit" and cache_result.agent_id:
                    logger.info("Routing cache hit: %s for '%s'", cache_result.agent_id, user_text[:80])
                    return cache_result.agent_id, user_text
            except Exception:
                logger.warning("Routing cache check failed, proceeding with LLM", exc_info=True)

        system_prompt = self._load_prompt("orchestrator")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        try:
            response = await self._call_llm(messages)
            target_agent, condensed = await self._parse_classification(response, user_text)
            # Store routing decision after successful LLM classification
            if self._cache_manager and target_agent != _FALLBACK_AGENT:
                try:
                    self._cache_manager.store_routing(user_text, target_agent, 1.0)
                except Exception:
                    logger.warning("Failed to store routing decision", exc_info=True)
            return target_agent, condensed
        except Exception:
            logger.exception("Intent classification failed, falling back to %s", _FALLBACK_AGENT)
            return _FALLBACK_AGENT, user_text

    async def _parse_classification(self, response: str, original_text: str) -> tuple[str, str]:
        """Parse LLM classification response.

        Expected format: "<agent-id>: <condensed task>"
        Falls back to general-agent if parsing fails.
        """
        response = response.strip()
        if ":" not in response:
            logger.warning("Could not parse classification: %s", response[:100])
            return _FALLBACK_AGENT, original_text

        agent_id, _, condensed = response.partition(":")
        agent_id = agent_id.strip().lower()
        condensed = condensed.strip()

        known_agents = await self._get_known_agents()
        if agent_id not in known_agents:
            logger.warning("Unknown agent '%s' in classification, falling back", agent_id)
            return _FALLBACK_AGENT, original_text

        if not condensed:
            condensed = original_text

        return agent_id, condensed

    def _get_turns(self, conversation_id: str | None) -> list[dict]:
        """Get recent conversation turns for context."""
        if not conversation_id:
            return []
        return list(self._conversations.get(conversation_id, []))

    def _store_turn(self, conversation_id: str | None, user_text: str, assistant_text: str) -> None:
        """Store a conversation turn, keeping last _MAX_TURNS exchanges."""
        if not conversation_id:
            return
        turns = self._conversations[conversation_id]
        turns.append({"role": "user", "content": user_text})
        turns.append({"role": "assistant", "content": assistant_text})
        # Keep only the last _MAX_TURNS * 2 messages (user+assistant pairs)
        max_messages = _MAX_TURNS * 2
        if len(turns) > max_messages:
            self._conversations[conversation_id] = turns[-max_messages:]
