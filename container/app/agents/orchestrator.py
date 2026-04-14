"""Orchestrator agent for intent classification and task dispatch."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

from app.agents.base import BaseAgent
from app.a2a.protocol import JsonRpcRequest
from app.db.repository import SettingsRepository
from app.agents.sanitize import strip_markdown
from app.agents.language_detect import detect_user_language
from app.analytics.collector import track_request, track_agent_timeout
from app.models.agent import AgentCard, AgentTask, TaskContext
from app.models.cache import ResponseCacheEntry, CachedAction

logger = logging.getLogger(__name__)

# Max conversation turns to keep per conversation
_MAX_TURNS = 3

# Conversation memory limits
_MAX_CONVERSATIONS = 1000
_CONVERSATION_TTL_SECONDS = 1800  # 30 minutes

# Fallback agent when classification fails
_FALLBACK_AGENT = "general-agent"


class _NoOpSpan:
    """No-op span for when span_collector is None."""

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return {}

    def get(self, key, default=None):
        return default


@contextlib.asynccontextmanager
async def _optional_span(span_collector, name, **kwargs):
    if span_collector:
        async with span_collector.start_span(name, **kwargs) as span:
            yield span
    else:
        yield _NoOpSpan()


class OrchestratorAgent(BaseAgent):
    """Classifies user intent and dispatches to specialized agents via A2A."""

    def __init__(self, dispatcher, registry=None, cache_manager=None, ha_client=None, entity_index=None, presence_detector=None, filler_agent=None) -> None:
        super().__init__(ha_client=ha_client, entity_index=entity_index)
        self._dispatcher = dispatcher
        self._registry = registry
        self._cache_manager = cache_manager
        self._presence_detector = presence_detector
        self._filler_agent = filler_agent
        self._conversations: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._default_timeout: int = 5
        self._max_iterations: int = 3
        self._filler_enabled: bool = False
        self._filler_threshold_ms: int = 1000
        self._mediation_model: str | None = None
        self._mediation_temperature: float = 0.3
        self._mediation_max_tokens: int = 2048

    async def initialize(self) -> None:
        """Load reliability config from DB. Call during startup."""
        await self._load_reliability_config()
        await self._load_filler_config()
        await self._load_mediation_config()

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

    async def _load_filler_config(self) -> None:
        """Read filler/interim response settings from DB."""
        try:
            val = await SettingsRepository.get_value("filler.enabled", "false")
            self._filler_enabled = val.lower() == "true"
        except (ValueError, TypeError):
            self._filler_enabled = False
        try:
            val = await SettingsRepository.get_value("filler.threshold_ms", "1000")
            self._filler_threshold_ms = int(val)
        except (ValueError, TypeError):
            self._filler_threshold_ms = 1000
        logger.info(
            "Filler config: enabled=%s threshold=%dms",
            self._filler_enabled, self._filler_threshold_ms,
        )

    async def _load_mediation_config(self) -> None:
        """Read mediation/merge override params from settings."""
        try:
            val = await SettingsRepository.get_value("mediation.model", "")
            self._mediation_model = val if val else None
        except (ValueError, TypeError):
            self._mediation_model = None
        try:
            val = await SettingsRepository.get_value("mediation.temperature", "0.3")
            self._mediation_temperature = float(val)
        except (ValueError, TypeError):
            self._mediation_temperature = 0.3
        try:
            val = await SettingsRepository.get_value("mediation.max_tokens", "2048")
            self._mediation_max_tokens = int(val)
        except (ValueError, TypeError):
            self._mediation_max_tokens = 2048
        logger.info(
            "Mediation config: model=%s temperature=%.1f max_tokens=%d",
            self._mediation_model or "(orchestrator default)",
            self._mediation_temperature,
            self._mediation_max_tokens,
        )

    async def _get_known_agents(self) -> set[str]:
        """Return set of currently registered agent IDs (excluding orchestrator)."""
        if not self._registry:
            return {"light-agent", "music-agent", "general-agent"}
        cards = await self._registry.list_agents()
        return {card.agent_id for card in cards if card.agent_id != "orchestrator"}

    async def _resolve_language(self, user_text: str, context_language: str | None = None, turns: list[dict] | None = None) -> str:
        """Resolve effective language: DB setting > auto-detect > turns-detect > fallback."""
        setting = await SettingsRepository.get_value("language", "auto")
        if setting and setting != "auto":
            return setting  # Manual override from settings
        # Auto-detect from user text
        detected = detect_user_language(user_text, fallback="")
        if detected:
            return detected
        # Low confidence on short text - try with recent conversation context
        if turns:
            user_turns = [t.get("content", "") for t in turns if t.get("role") == "user"]
            if user_turns:
                combined = " ".join(user_turns[-3:]) + " " + user_text
                detected = detect_user_language(combined, fallback="")
                if detected:
                    return detected
        return context_language or "en"

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id="orchestrator",
            name="Orchestrator",
            description="Routes user requests to the appropriate specialized agent.",
            skills=["intent_classification", "task_routing"],
            endpoint="local://orchestrator",
        )

    async def _dispatch_single(
        self,
        target_agent: str,
        condensed_task: str,
        user_text: str,
        conversation_id: str | None,
        turns: list[dict],
        span_collector,
        incoming_context: TaskContext | None = None,
        skip_dispatch_span: bool = False,
    ) -> tuple[str, str, dict | None]:
        """Dispatch a single task to one agent and return (agent_id, speech, result_dict)."""
        context = TaskContext(conversation_turns=turns)
        if self._presence_detector:
            room = self._presence_detector.get_most_likely_room()
            if room:
                context.presence_room = room
        if incoming_context:
            context.device_id = incoming_context.device_id
            context.area_id = incoming_context.area_id
            context.language = incoming_context.language

        agent_task = AgentTask(
            description=condensed_task,
            user_text=user_text,
            conversation_id=conversation_id,
            context=context,
        )
        request = JsonRpcRequest(
            method="message/send",
            params={
                "agent_id": target_agent,
                "task": agent_task.model_dump(),
                "_span_collector": span_collector,
            },
            id=conversation_id or "orchestrator-dispatch",
        )
        try:
            t0 = time.perf_counter()
            noop_span = {"metadata": {}}
            dispatch_ctx = (
                contextlib.nullcontext(noop_span)
                if skip_dispatch_span
                else _optional_span(span_collector, "dispatch", agent_id=target_agent)
            )
            async with dispatch_ctx as span:
                response = await asyncio.wait_for(
                    self._dispatcher.dispatch(request),
                    timeout=self._default_timeout,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                span["metadata"]["latency_ms"] = round(latency_ms, 1)
                result_data = response.result or {}
                span["metadata"]["agent_response"] = (result_data.get("speech") or "")[:500]
                span["metadata"]["condensed_task"] = condensed_task[:500]
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
                return target_agent, "I couldn't process that request in time.", None

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

        # Log structured errors from agents for observability
        error = result.get("error")
        if error:
            error_code = error.get("code", "unknown")
            logger.info(
                "Agent %s returned error: %s (recoverable=%s)",
                target_agent, error_code, error.get("recoverable", True),
            )

        return target_agent, speech, result

    async def _handle_sequential_send(
        self,
        classifications: list[tuple[str, str, float]],
        user_text: str,
        conversation_id: str,
        turns: list[dict],
        span_collector,
        incoming_context,
    ) -> tuple[str, str, dict | None]:
        """Handle sequential dispatch: content agent -> send agent.

        Returns (routed_to, speech, result_dict) like _dispatch_single.
        """
        content_agents = [(a, t, c) for a, t, c in classifications if a != "send-agent"]
        send_classification = next(((a, t, c) for a, t, c in classifications if a == "send-agent"), None)

        if not send_classification:
            logger.warning("_handle_sequential_send called without send-agent classification")
            return await self._dispatch_single(
                classifications[0][0], classifications[0][1], user_text,
                conversation_id, turns, span_collector, incoming_context=incoming_context,
            )

        _send_agent_id, send_task_text, _send_confidence = send_classification

        # Step 1: Dispatch content agent(s)
        if content_agents:
            content_aid, content_task, _ = content_agents[0]
            content_context = TaskContext(
                conversation_turns=turns,
                device_id=incoming_context.device_id if incoming_context else None,
                area_id=incoming_context.area_id if incoming_context else None,
                language=incoming_context.language if incoming_context else "en",
                sequential_send=True,
            )
            if self._presence_detector:
                room = self._presence_detector.get_most_likely_room()
                if room:
                    content_context.presence_room = room
            async with _optional_span(span_collector, "dispatch_content", agent_id=content_aid) as span:
                content_agent_id, content_speech, _content_result = await self._dispatch_single(
                    content_aid, content_task, user_text, conversation_id, turns, span_collector,
                    incoming_context=content_context,
                    skip_dispatch_span=True,
                )
                span["metadata"]["content_agent"] = content_agent_id
                span["metadata"]["content_length"] = len(content_speech or "")
                span["metadata"]["agent_response"] = (content_speech or "")[:500]
                span["metadata"]["condensed_task"] = content_task[:500]
        else:
            content_speech = turns[-1].get("content", "") if turns else ""
            content_agent_id = "conversation-history"

        if not content_speech:
            return (
                "send-agent",
                "No content available to send.",
                {"speech": "No content available to send.", "error": {
                    "code": "parse_error", "recoverable": True,
                }},
            )

        # Step 2: Build augmented task for send-agent with the content
        from app.agents.send import _CONTENT_SEPARATOR
        augmented_task = f"{send_task_text}{_CONTENT_SEPARATOR}{content_speech}"

        async with _optional_span(span_collector, "dispatch_send", agent_id="send-agent") as span:
            send_aid, send_speech, send_result = await self._dispatch_single(
                "send-agent", augmented_task, user_text, conversation_id, turns, span_collector,
                incoming_context=incoming_context,
                skip_dispatch_span=True,
            )
            span["metadata"]["send_target"] = send_task_text
            span["metadata"]["content_from"] = content_agent_id
            span["metadata"]["agent_response"] = (send_speech or "")[:500]
            span["metadata"]["condensed_task"] = augmented_task[:500]

        routed_to = f"{content_agent_id}, send-agent"

        return routed_to, send_speech, send_result

    # ------------------------------------------------------------------
    # Shared helpers to reduce duplication between handle_task / handle_task_stream
    # ------------------------------------------------------------------

    async def _do_cache_lookup(self, user_text: str, span_collector) -> tuple:
        """Perform cache lookup with optional span recording.

        Returns (cache_result, cache_span_or_None).
        """
        cache_result = None
        cache_span_ref = None
        if not self._cache_manager:
            return cache_result, cache_span_ref

        async with _optional_span(span_collector, "cache_lookup", agent_id="orchestrator") as cache_span:
            try:
                cache_result = await self._cache_manager.process(user_text)
            except Exception:
                logger.warning("Cache lookup failed", exc_info=True)
                cache_result = None
            if cache_result:
                cache_span["metadata"]["hit_type"] = cache_result.hit_type
                cache_span["metadata"]["similarity"] = cache_result.similarity
                cache_span["metadata"]["cached_agent_id"] = cache_result.agent_id
                if cache_result.hit_type.startswith("response"):
                    cache_span["metadata"]["cache_tier"] = "response"
                elif cache_result.hit_type == "routing_hit":
                    cache_span["metadata"]["cache_tier"] = "routing"
                else:
                    cache_span["metadata"]["cache_tier"] = "both_miss"
                if cache_result.entry:
                    cache_span["metadata"]["hit_count"] = getattr(cache_result.entry, "hit_count", None)
                    created = getattr(cache_result.entry, "created_at", None)
                    if created:
                        try:
                            from datetime import datetime, timezone
                            created_dt = datetime.fromisoformat(created)
                            age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                            cache_span["metadata"]["entry_age_seconds"] = round(age, 1)
                        except Exception:
                            pass
            else:
                cache_span["metadata"]["hit_type"] = "miss"
                cache_span["metadata"]["cache_tier"] = "both_miss"
            cache_span_ref = cache_span

        return cache_result, cache_span_ref

    async def _handle_response_cache_hit(
        self,
        cache_result,
        conversation_id: str,
        user_text: str,
        span_collector,
        cache_span=None,
    ) -> dict:
        """Handle a response cache hit: execute cached action, rewrite, store turn, trace.

        Returns the response dict.
        """
        target_agent = cache_result.agent_id or "unknown"
        action_executed = None

        async def _do_ha():
            if not cache_result.cached_action:
                return None
            try:
                return await self._execute_cached_action(cache_result.cached_action)
            except Exception:
                logger.warning("Cached action execution failed", exc_info=True)
                return None

        async def _do_rewrite():
            if self._cache_manager:
                await self._cache_manager.apply_rewrite(cache_result)

        ha_result, _ = await asyncio.gather(_do_ha(), _do_rewrite())
        action_executed = ha_result
        speech = cache_result.response_text or ""

        if cache_result.cached_action:
            async with _optional_span(span_collector, "ha_action", agent_id=target_agent) as ha_span:
                ha_span["metadata"]["action"] = cache_result.cached_action.service
                ha_span["metadata"]["entity"] = cache_result.cached_action.entity_id
                ha_span["metadata"]["success"] = action_executed is not None
                ha_span["metadata"]["cached"] = True
            if cache_span:
                try:
                    from datetime import datetime, timedelta
                    _cs = datetime.fromisoformat(cache_span["start_time"])
                    ha_span["start_time"] = (
                        _cs + timedelta(milliseconds=cache_span.get("duration_ms", 0))
                    ).isoformat()
                except Exception:
                    pass
        if cache_result.rewrite_applied:
            async with _optional_span(span_collector, "rewrite", agent_id="rewrite-agent") as rw_span:
                rw_span["metadata"]["original_text"] = (cache_result.original_response_text or "")[:500]
                rw_span["metadata"]["rewritten_text"] = speech[:500]
                rw_span["metadata"]["latency_ms"] = cache_result.rewrite_latency_ms
                rw_span["metadata"]["success"] = True
            if cache_result.rewrite_latency_ms is not None:
                rw_span["duration_ms"] = round(cache_result.rewrite_latency_ms, 2)
            if cache_span:
                try:
                    from datetime import datetime, timedelta
                    _cs = datetime.fromisoformat(cache_span["start_time"])
                    rw_span["start_time"] = (
                        _cs + timedelta(milliseconds=cache_span.get("duration_ms", 0))
                    ).isoformat()
                except Exception:
                    pass
        async with _optional_span(span_collector, "return", agent_id="orchestrator") as ret_span:
            ret_span["metadata"]["from_agent"] = target_agent
            ret_span["metadata"]["agent_response"] = speech[:500]
            ret_span["metadata"]["final_response"] = speech[:500]
            ret_span["metadata"]["mediated"] = False
            ret_span["metadata"]["response_cache_hit"] = True
            prior_turns = self._get_turns(conversation_id)
            self._store_turn(conversation_id, user_text, speech, agent_id=target_agent)
            if span_collector:
                try:
                    from app.analytics.tracer import create_trace_summary
                    await create_trace_summary(
                        trace_id=span_collector.trace_id,
                        conversation_id=conversation_id,
                        user_input=user_text,
                        final_response=speech,
                        routing_agent=target_agent,
                        routing_confidence=1.0,
                        routing_duration_ms=None,
                        condensed_task=user_text,
                        agents=["orchestrator", target_agent],
                        source=getattr(span_collector, "source", "api"),
                        conversation_turns=prior_turns,
                    )
                except Exception:
                    logger.warning("Failed to create trace summary", exc_info=True)

        return {
            "speech": speech,
            "routed_to": target_agent,
            "action_executed": action_executed,
        }

    def _store_response_cache(
        self,
        user_text: str,
        speech: str,
        target_agent: str,
        confidence: float,
        action_executed,
        has_error: bool,
    ) -> bool:
        """Store a response in the cache. Returns True if stored."""
        if (
            not self._cache_manager
            or target_agent == _FALLBACK_AGENT
            or not speech
            or has_error
        ):
            return False
        try:
            cached_action = None
            entity_ids: list[str] = []
            if action_executed and action_executed.get("success"):
                entity_id_str = action_executed.get("entity_id", "")
                domain = entity_id_str.split(".")[0] if "." in entity_id_str else ""
                cached_action = CachedAction(
                    service=f"{domain}/{action_executed['action']}",
                    entity_id=entity_id_str,
                    service_data={},
                )
                entity_ids = [entity_id_str] if entity_id_str else []
            entry = ResponseCacheEntry(
                query_text=user_text,
                response_text=speech,
                agent_id=target_agent,
                cached_action=cached_action,
                confidence=confidence,
                entity_ids=entity_ids,
            )
            self._cache_manager.store_response(entry)
            return True
        except Exception:
            logger.warning("Failed to store response cache", exc_info=True)
            return False

    async def _create_trace(
        self,
        span_collector,
        conversation_id: str,
        user_text: str,
        speech: str,
        target_agent: str,
        confidence: float,
        condensed_task: str,
        classifications: list[tuple[str, str, float]],
        turns: list[dict],
    ) -> None:
        """Create a trace summary from span data."""
        try:
            from app.analytics.tracer import create_trace_summary
            classify_duration = None
            for s in span_collector._spans:
                if s.get("span_name") == "classify":
                    classify_duration = s.get("duration_ms")
                    break
            agents = list({s.get("agent_id") for s in span_collector._spans if s.get("agent_id")})
            if "orchestrator" not in agents:
                agents.insert(0, "orchestrator")
            await create_trace_summary(
                trace_id=span_collector.trace_id,
                conversation_id=conversation_id,
                user_input=user_text,
                final_response=speech,
                routing_agent=target_agent,
                routing_confidence=confidence,
                routing_duration_ms=classify_duration,
                condensed_task=condensed_task,
                agents=agents,
                source=getattr(span_collector, "source", "api"),
                agent_instructions={aid: ctask for aid, ctask, _ in classifications} if len(classifications) > 1 else None,
                conversation_turns=turns,
            )
        except Exception:
            logger.warning("Failed to create trace summary", exc_info=True)

    async def handle_task(self, task: AgentTask, *, _pre_classified: tuple[list[tuple[str, str, float]], bool] | None = None) -> dict:
        user_text = task.user_text or task.description
        conversation_id = task.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.debug("No conversation_id from HA, generated fallback: %s", conversation_id)

        # Resolve effective language via DB setting / auto-detect
        context_language = (task.context.language if task.context else None) or "en"
        lang_turns = self._get_turns(conversation_id)
        detected_language = await self._resolve_language(user_text, context_language, turns=lang_turns)

        # Get span collector from task context if available
        span_collector = task.span_collector

        # 0. Cache lookup (before classify)
        cache_result = None
        cache_span_ref = None
        if _pre_classified is None and self._cache_manager:
            cache_result, cache_span_ref = await self._do_cache_lookup(user_text, span_collector)

        # Response hit short-circuit: skip classify and dispatch entirely
        if cache_result and cache_result.hit_type == "response_hit":
            result = await self._handle_response_cache_hit(
                cache_result, conversation_id, user_text, span_collector, cache_span_ref,
            )
            return result

        # 1. Classify intent (skip if pre-classified by handle_task_stream)
        if _pre_classified is not None:
            classifications, routing_cached = _pre_classified
            target_agent, condensed_task, confidence = classifications[0]
        else:
            async with _optional_span(span_collector, "classify", agent_id="orchestrator") as span:
                classifications, routing_cached = await self._classify(user_text, cache_result=cache_result, conversation_id=conversation_id)
                target_agent, condensed_task, confidence = classifications[0]
                span["metadata"]["target_agent"] = ", ".join(a for a, _, _ in classifications)
                span["metadata"]["user_input"] = user_text[:500]
                span["metadata"]["condensed_task"] = condensed_task[:500]
                span["metadata"]["confidence"] = confidence
                span["metadata"]["routing_cached"] = routing_cached
                span["metadata"]["multi_agent"] = len(classifications) > 1
                if len(classifications) > 1:
                    span["metadata"]["all_classifications"] = {
                        a: {"task": t[:300], "confidence": c}
                        for a, t, c in classifications
                    }
        logger.info(
            "Routed to %s (%.0f%%): %s (conversation=%s)",
            target_agent, confidence * 100, condensed_task[:80], conversation_id,
        )

        # 2. Build context with conversation turns
        turns = self._get_turns(conversation_id)

        # Check for sequential send dispatch
        is_sequential_send = (
            len(classifications) > 1
            and any(a == "send-agent" for a, _, _ in classifications)
        )

        # 3-4. Dispatch
        incoming_context = task.context
        if is_sequential_send:
            # --- Sequential send dispatch (content agent -> send agent) ---
            routed_to, speech, result = await self._handle_sequential_send(
                classifications, user_text, conversation_id, turns, span_collector,
                incoming_context=incoming_context,
            )
            action_executed = (result or {}).get("action_executed")
            has_error = bool((result or {}).get("error"))
            agent_error = (result or {}).get("error")
        elif len(classifications) == 1:
            # --- Single agent path (unchanged flow) ---
            agent_id, speech, result = await self._dispatch_single(
                target_agent, condensed_task, user_text, conversation_id, turns, span_collector,
                incoming_context=incoming_context,
            )
            action_executed = (result or {}).get("action_executed")
            routed_to = agent_id

            # Check for agent-level error
            agent_error = (result or {}).get("error")
            has_error = agent_error is not None
        else:
            # --- Multi-agent parallel dispatch ---
            dispatch_coros = [
                self._dispatch_single(aid, ctask, user_text, conversation_id, turns, span_collector,
                                      incoming_context=incoming_context)
                for aid, ctask, _ in classifications
            ]
            dispatch_results = await asyncio.gather(*dispatch_coros, return_exceptions=True)

            agent_responses: list[tuple[str, str, bool]] = []
            action_executed = None
            routed_agents: list[str] = []
            for dr in dispatch_results:
                if isinstance(dr, Exception):
                    logger.warning("Multi-agent dispatch error: %s", dr)
                    continue
                aid, sp, res = dr
                routed_agents.append(aid)
                acted = bool(res and res.get("action_executed"))
                agent_responses.append((aid, sp, acted))
                if res and res.get("action_executed") and action_executed is None:
                    action_executed = res["action_executed"]

            target_agent = routed_agents[0] if routed_agents else _FALLBACK_AGENT
            routed_to = ", ".join(routed_agents) if routed_agents else _FALLBACK_AGENT
            speech = ""  # Will be set by _merge_responses inside return span
            has_error = False

        # 5. Mediate response, store turn, trace
        original_speech = speech
        cache_stored_response = False
        async with _optional_span(span_collector, "return", agent_id="orchestrator") as ret_span:
            ret_span["metadata"]["from_agent"] = routed_to
            if len(classifications) > 1 and not is_sequential_send:
                speech = await self._merge_responses(agent_responses, user_text)
                result = {"speech": speech}
            ret_span["metadata"]["agent_response"] = speech[:500]
            if (len(classifications) <= 1 or is_sequential_send) and not has_error:
                mediation_agent = "send-agent" if is_sequential_send else target_agent
                mediation_language = detected_language
                speech = await self._mediate_response(speech, user_text, mediation_agent, language=mediation_language)
            ret_span["metadata"]["final_response"] = speech[:500]
            ret_span["metadata"]["mediated"] = (speech != original_speech) or len(classifications) > 1
            if len(classifications) == 1:
                cache_stored_response = self._store_response_cache(
                    user_text, speech, target_agent, confidence, action_executed, has_error,
                )
            ret_span["metadata"]["cache_stored_response"] = cache_stored_response
            self._store_turn(conversation_id, user_text, speech, agent_id=routed_to)
            if span_collector:
                await self._create_trace(
                    span_collector, conversation_id, user_text, speech,
                    target_agent, confidence, condensed_task, classifications, turns,
                )

        response = {
            "speech": strip_markdown(speech),
            "conversation_id": conversation_id,
            "routed_to": routed_to,
            "action_executed": action_executed,
        }
        if has_error:
            response["error"] = {
                "code": agent_error.get("code", "unknown") if agent_error else "unknown",
                "recoverable": agent_error.get("recoverable", True) if agent_error else True,
            }
        return response

    async def handle_task_stream(self, task: AgentTask) -> AsyncGenerator[dict, None]:
        user_text = task.user_text or task.description
        conversation_id = task.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.debug("No conversation_id from HA, generated fallback: %s", conversation_id)

        # Resolve effective language via DB setting / auto-detect
        context_language = (task.context.language if task.context else None) or "en"
        lang_turns = self._get_turns(conversation_id)
        detected_language = await self._resolve_language(user_text, context_language, turns=lang_turns)

        span_collector = task.span_collector
        t0_request = time.perf_counter()  # Wall-clock start for filler threshold
        t0_request_utc = datetime.now(timezone.utc)  # Absolute UTC for span timestamp overrides

        # 0. Cache lookup (before classify)
        cache_result = None
        cache_span_ref = None
        if self._cache_manager:
            cache_result, cache_span_ref = await self._do_cache_lookup(user_text, span_collector)

        # Response hit short-circuit for streaming
        if cache_result and cache_result.hit_type == "response_hit":
            result = await self._handle_response_cache_hit(
                cache_result, conversation_id, user_text, span_collector, cache_span_ref,
            )
            yield {
                "token": strip_markdown(result["speech"]),
                "done": True,
                "conversation_id": conversation_id,
            }
            return

        # 1. Classify (non-streaming -- fast via Groq)
        async with _optional_span(span_collector, "classify", agent_id="orchestrator") as span:
            classifications, routing_cached = await self._classify(user_text, cache_result=cache_result, conversation_id=conversation_id)
            target_agent, condensed_task, confidence = classifications[0]
            span["metadata"]["target_agent"] = ", ".join(a for a, _, _ in classifications)
            span["metadata"]["user_input"] = user_text[:500]
            span["metadata"]["condensed_task"] = condensed_task[:500]
            span["metadata"]["confidence"] = confidence
            span["metadata"]["routing_cached"] = routing_cached
            span["metadata"]["multi_agent"] = len(classifications) > 1
        logger.info(
            "Stream routed to %s: %s (conversation=%s)",
            target_agent, condensed_task[:80], conversation_id,
        )

        # Multi-agent: yield progress marker, then fall back to non-streaming handle_task
        is_sequential_send = (
            len(classifications) > 1
            and any(a == "send-agent" for a, _, _ in classifications)
        )

        # Sequential send: fall back to non-streaming, with filler support
        if is_sequential_send:
            yield {
                "token": "",
                "done": False,
                "conversation_id": conversation_id,
                "status": "sequential_send",
            }

            # Determine which content agent to check for filler
            content_agent_ids = [a for a, _, _ in classifications if a != "send-agent"]
            content_agent_for_filler = content_agent_ids[0] if content_agent_ids else None
            seq_use_filler = (
                await self._should_send_filler(content_agent_for_filler)
                if content_agent_for_filler
                else False
            )
            language = detected_language

            seq_filler_sent = False
            seq_filler_text = ""
            seq_filler_start_ms = 0.0
            seq_filler_end_ms = 0.0
            seq_filler_generated = False
            seq_filler_send_ms = 0.0

            if seq_use_filler:
                # Race handle_task against filler threshold
                task_coro = self.handle_task(task, _pre_classified=(classifications, routing_cached))
                task_future = asyncio.ensure_future(task_coro)

                elapsed = time.perf_counter() - t0_request
                remaining = max(0, self._filler_threshold_ms / 1000 - elapsed)

                done_set, _ = await asyncio.wait({task_future}, timeout=remaining)
                if done_set:
                    # handle_task completed before threshold -- no filler needed
                    result = task_future.result()
                else:
                    # Threshold exceeded -- generate filler
                    seq_filler_start_ms = (time.perf_counter() - t0_request) * 1000
                    filler_text = await self._invoke_filler_agent(
                        user_text, content_agent_for_filler, language,
                    )
                    seq_filler_end_ms = (time.perf_counter() - t0_request) * 1000

                    if filler_text and not task_future.done():
                        seq_filler_generated = True
                        seq_filler_text = filler_text
                        seq_filler_send_ms = (time.perf_counter() - t0_request) * 1000
                        yield {
                            "token": filler_text,
                            "done": False,
                            "is_filler": True,
                            "conversation_id": conversation_id,
                        }
                        seq_filler_sent = True
                    elif filler_text:
                        seq_filler_generated = True
                        seq_filler_text = filler_text

                    result = await task_future
            else:
                result = await self.handle_task(task, _pre_classified=(classifications, routing_cached))

            # Record filler_generate span
            if seq_filler_generated:
                async with _optional_span(span_collector, "filler_generate", agent_id="filler-agent") as fg_span:
                    fg_span["metadata"]["threshold_ms"] = self._filler_threshold_ms
                    fg_span["metadata"]["target_agent"] = content_agent_for_filler
                    fg_span["metadata"]["filler_text"] = seq_filler_text
                    fg_span["metadata"]["sequential_send"] = True
                    fg_span["metadata"]["was_sent"] = seq_filler_sent
                    if seq_filler_start_ms > 0:
                        actual_start = t0_request_utc + timedelta(milliseconds=seq_filler_start_ms)
                        fg_span["start_time"] = actual_start.isoformat()
                        fg_span["_override_duration_ms"] = round(
                            seq_filler_end_ms - seq_filler_start_ms, 2,
                        )

            # Record filler_send span
            if seq_filler_sent:
                async with _optional_span(span_collector, "filler_send", agent_id="filler-agent") as fs_span:
                    fs_span["metadata"]["target_agent"] = content_agent_for_filler
                    fs_span["metadata"]["filler_text"] = seq_filler_text
                    fs_span["metadata"]["sequential_send"] = True
                    if seq_filler_send_ms > 0:
                        actual_start = t0_request_utc + timedelta(milliseconds=seq_filler_send_ms)
                        fs_span["start_time"] = actual_start.isoformat()
                        fs_span["_override_duration_ms"] = 0

            yield {
                "token": result["speech"],
                "done": True,
                "conversation_id": conversation_id,
            }
            return

        if len(classifications) > 1:
            yield {
                "token": "",
                "done": False,
                "conversation_id": conversation_id,
                "status": "multi_agent",
                "agents": [a for a, _, _ in classifications],
            }
            result = await self.handle_task(task, _pre_classified=(classifications, routing_cached))
            yield {
                "token": result["speech"],
                "done": True,
                "conversation_id": conversation_id,
            }
            return

        # 2. Build context and task (single agent streaming)
        turns = self._get_turns(conversation_id)
        language = detected_language
        context = TaskContext(conversation_turns=turns, language=language)
        if task.context:
            context.device_id = task.context.device_id
            context.area_id = task.context.area_id
        agent_task = AgentTask(
            description=condensed_task,
            user_text=user_text,
            conversation_id=conversation_id,
            context=context,
        )

        # 3. Dispatch via A2A message/stream
        request = JsonRpcRequest(
            method="message/stream",
            params={
                "agent_id": target_agent,
                "task": agent_task.model_dump(),
                "_span_collector": span_collector,
            },
            id=conversation_id or "orchestrator-stream",
        )

        t0_dispatch = time.perf_counter()
        collected_speech = []
        action_executed = None
        use_filler = await self._should_send_filler(target_agent)
        filler_sent = False
        filler_text_sent = ""
        filler_start_ms = 0.0
        filler_end_ms = 0.0
        filler_generated = False
        filler_send_ms = 0.0
        logger.info("Filler decision for %s: use_filler=%s (enabled=%s)", target_agent, use_filler, self._filler_enabled)

        async def _process_chunk(chunk):
            """Process a single stream chunk: collect speech and detect actions."""
            nonlocal action_executed
            token = chunk.result.get("token", "")
            done = chunk.result.get("done", False)
            error = chunk.result.get("error")
            if error:
                logger.warning("Agent streaming error: %s", error)
            if token:
                collected_speech.append(token)
            if done and chunk.result.get("action_executed"):
                action_executed = chunk.result["action_executed"]
            return token

        async def _stream_with_filler(stream_iter, span=None):
            """Race the first agent token against the filler threshold.

            Uses an asyncio.Queue to decouple the async generator reader
            from the consumer, so cancellation on timeout does not corrupt
            the generator state.
            """
            nonlocal filler_sent, filler_text_sent, filler_start_ms, filler_end_ms, filler_generated, filler_send_ms

            if not use_filler:
                # No filler logic -- stream directly
                async for chunk in stream_iter:
                    token = await _process_chunk(chunk)
                    if token:
                        yield {
                            "token": token,
                            "done": False,
                            "conversation_id": None,
                        }
                return

            # Queue-based approach: reader task fills queue, main loop consumes
            queue: asyncio.Queue = asyncio.Queue()
            _sentinel = object()

            async def _reader():
                try:
                    async for chunk in stream_iter:
                        await queue.put(chunk)
                finally:
                    await queue.put(_sentinel)

            reader_task = asyncio.create_task(_reader())

            try:
                # Wait for first chunk or threshold (accounting for time already spent on classify)
                first_chunk = None
                elapsed_since_request = time.perf_counter() - t0_request
                remaining_threshold = max(0, self._filler_threshold_ms / 1000 - elapsed_since_request)
                logger.info("Filler remaining threshold: %.1fms (elapsed %.0fms)", remaining_threshold * 1000, elapsed_since_request * 1000)
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=remaining_threshold,
                    )
                    logger.info("First chunk arrived before threshold")
                    if item is not _sentinel:
                        first_chunk = item
                except asyncio.TimeoutError:
                    # Agent is slow -- generate and yield filler
                    logger.info("Threshold exceeded, generating filler for %s", target_agent)
                    filler_start_ms = (time.perf_counter() - t0_request) * 1000
                    filler_text = await self._invoke_filler_agent(user_text, target_agent, language)
                    filler_end_ms = (time.perf_counter() - t0_request) * 1000
                    logger.info("Filler generation result: %s", repr(filler_text[:80]) if filler_text else "None")
                    if filler_text:
                        filler_generated = True
                        filler_text_sent = filler_text
                        # Check if agent already responded during filler generation
                        if not queue.empty():
                            logger.info("Agent responded during filler generation, skipping filler")
                        else:
                            filler_send_ms = (time.perf_counter() - t0_request) * 1000
                            yield {
                                "token": filler_text,
                                "done": False,
                                "is_filler": True,
                                "conversation_id": None,
                            }
                            filler_sent = True
                            logger.info("Filler sent for %s: %s", target_agent, filler_text[:80])
                    # Now wait for the real first chunk
                    item = await queue.get()
                    if item is not _sentinel:
                        first_chunk = item

                # Process first chunk
                if first_chunk is not None:
                    token = await _process_chunk(first_chunk)
                    if token:
                        yield {
                            "token": token,
                            "done": False,
                            "conversation_id": None,
                        }

                # Drain remaining chunks from queue
                while True:
                    item = await queue.get()
                    if item is _sentinel:
                        break
                    token = await _process_chunk(item)
                    if token:
                        yield {
                            "token": token,
                            "done": False,
                            "conversation_id": None,
                        }
            finally:
                await reader_task

        async with _optional_span(span_collector, "dispatch", agent_id=target_agent) as span:
            async for token_dict in _stream_with_filler(self._dispatcher.dispatch_stream(request), span):
                yield token_dict
            span["metadata"]["token_count"] = len(collected_speech)
            span["metadata"]["agent_response"] = "".join(collected_speech)[:500]
            if filler_sent:
                span["metadata"]["filler_sent"] = True

        latency_ms = (time.perf_counter() - t0_dispatch) * 1000
        await track_request(target_agent, cache_hit=False, latency_ms=latency_ms)

        # Record filler_generate span (always, if filler was generated -- even if not sent)
        if filler_generated:
            async with _optional_span(span_collector, "filler_generate", agent_id="filler-agent") as fg_span:
                fg_span["metadata"]["threshold_ms"] = self._filler_threshold_ms
                fg_span["metadata"]["target_agent"] = target_agent
                fg_span["metadata"]["filler_text"] = filler_text_sent
                fg_span["metadata"]["was_sent"] = filler_sent
                if filler_start_ms > 0:
                    actual_start = t0_request_utc + timedelta(milliseconds=filler_start_ms)
                    fg_span["start_time"] = actual_start.isoformat()
                    fg_span["_override_duration_ms"] = round(filler_end_ms - filler_start_ms, 2)

        # Record filler_send span (only if filler was actually yielded to user)
        if filler_sent:
            async with _optional_span(span_collector, "filler_send", agent_id="filler-agent") as fs_span:
                fs_span["metadata"]["target_agent"] = target_agent
                fs_span["metadata"]["filler_text"] = filler_text_sent
                if filler_send_ms > 0:
                    actual_start = t0_request_utc + timedelta(milliseconds=filler_send_ms)
                    fs_span["start_time"] = actual_start.isoformat()
                    fs_span["_override_duration_ms"] = 0

        # 4. Store conversation turn and create trace summary
        full_speech = "".join(collected_speech)
        has_error = False  # streaming path does not propagate agent errors yet
        async with _optional_span(span_collector, "return", agent_id="orchestrator") as ret_span:
            ret_span["metadata"]["from_agent"] = target_agent
            ret_span["metadata"]["agent_response"] = full_speech[:500]
            full_speech = await self._mediate_response(full_speech, user_text, target_agent, language=language)
            ret_span["metadata"]["final_response"] = full_speech[:500]
            ret_span["metadata"]["mediated"] = (full_speech != "".join(collected_speech))
            cache_stored_response = self._store_response_cache(
                user_text, full_speech, target_agent, confidence, action_executed, has_error,
            )
            ret_span["metadata"]["cache_stored_response"] = cache_stored_response
            self._store_turn(conversation_id, user_text, full_speech, agent_id=target_agent)
            if span_collector:
                classifications = [(target_agent, condensed_task, confidence)]
                await self._create_trace(
                    span_collector, conversation_id, user_text, full_speech,
                    target_agent, confidence, condensed_task, classifications, turns,
                )

        # Yield final done chunk with optional mediated_speech
        mediated_text = strip_markdown(full_speech)
        raw_text = "".join(collected_speech)
        final_chunk = {
            "token": "",
            "done": True,
            "conversation_id": conversation_id,
        }
        if mediated_text != strip_markdown(raw_text):
            final_chunk["mediated_speech"] = mediated_text
        yield final_chunk

    async def _should_send_filler(self, target_agent: str) -> bool:
        """Check if filler is enabled and the target agent is expected to be slow."""
        if not self._filler_enabled:
            return False
        if not self._registry:
            return False
        cards = await self._registry.list_agents()
        for card in cards:
            if card.agent_id == target_agent:
                return card.expected_latency == "high"
        return False

    async def _invoke_filler_agent(self, user_text: str, target_agent: str, language: str) -> str | None:
        """Call the filler-agent directly to generate a filler phrase.

        Returns the filler text or None if generation fails.
        """
        if not self._filler_agent:
            return None
        try:
            context = TaskContext(language=language)
            filler_task = AgentTask(
                description=f"generate_filler:{target_agent}",
                user_text=user_text,
                context=context,
            )
            result = await self._filler_agent.handle_task(filler_task)
            speech = result.speech if hasattr(result, "speech") else (result or {}).get("speech", "")
            return speech.strip() if speech else None
        except Exception:
            logger.warning("Filler agent invocation failed", exc_info=True)
            return None

    async def _execute_cached_action(self, cached_action) -> dict | None:
        """Execute a cached action via HA client. Returns action result or None."""
        if not self._ha_client or not cached_action:
            return None
        try:
            service = cached_action.service or ""
            if "/" in service:
                domain, action = service.split("/", 1)
            else:
                domain, action = service, ""
            result = await self._ha_client.call_service(
                domain,
                action,
                entity_id=cached_action.entity_id,
                service_data=cached_action.service_data or None,
            )
            return result
        except Exception:
            logger.warning("Cached action execution failed", exc_info=True)
            return None

    async def _build_agent_descriptions(self) -> str:
        """Build agent list for classification prompt from registered AgentCards."""
        if not self._registry:
            return "- general-agent: fallback for general questions and unroutable requests"

        cards = await self._registry.list_agents()
        lines = []
        for card in cards:
            if card.agent_id == "orchestrator":
                continue
            if card.agent_id == "rewrite-agent":
                continue
            skills_str = ", ".join(card.skills) if card.skills else ""
            if skills_str:
                lines.append(f"- {card.agent_id}: {card.description} (skills: {skills_str})")
            else:
                lines.append(f"- {card.agent_id}: {card.description}")
        if not lines:
            lines.append("- general-agent: fallback for general questions and unroutable requests")
        return "\n".join(lines)

    async def _classify(self, user_text: str, *, cache_result=None, conversation_id: str | None = None) -> tuple[list[tuple[str, str, float]], bool]:
        """Classify user intent and produce a condensed task.

        The condensed task is a clear, actionable English description of
        what the agent should do. All entity/device/room/location names
        from the user's original text are preserved EXACTLY (verbatim,
        never translated or normalized).

        Args:
            user_text: The raw user input.
            cache_result: Optional pre-computed CacheResult from handle_task.

        Returns:
            (classifications, routing_cached) where classifications is a list
            of (target_agent_id, condensed_task, confidence) tuples.
        """
        # Use pre-computed cache result if available (avoids double lookup)
        if cache_result is not None:
            if cache_result.hit_type == "routing_hit" and cache_result.agent_id:
                logger.info("Routing cache hit: %s for '%s'", cache_result.agent_id, user_text[:80])
                condensed = cache_result.condensed_task or user_text
                return [(cache_result.agent_id, condensed, 1.0)], True
        elif self._cache_manager:
            # Fallback: no pre-computed result (e.g. called without handle_task)
            try:
                cache_result = await self._cache_manager.process(user_text)
                if cache_result.hit_type == "routing_hit" and cache_result.agent_id:
                    logger.info("Routing cache hit: %s for '%s'", cache_result.agent_id, user_text[:80])
                    condensed = cache_result.condensed_task or user_text
                    return [(cache_result.agent_id, condensed, 1.0)], True
            except Exception:
                logger.warning("Routing cache check failed, proceeding with LLM", exc_info=True)

        system_prompt_template = self._load_prompt("orchestrator")
        agent_descriptions = await self._build_agent_descriptions()
        system_prompt = system_prompt_template.replace("{agent_descriptions}", agent_descriptions)
        if "send-agent" not in agent_descriptions:
            system_prompt = self._strip_seq_rule(system_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Inject recent conversation history as proper multi-turn messages
        turns = self._get_turns(conversation_id)
        if turns:
            for turn in turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        try:
            response = await self._call_llm(messages)
            logger.info("Classification LLM response for '%s': %s", user_text[:60], repr(response[:300]))
            classifications = await self._parse_classification(response, user_text)
            # Store routing decision only for single-agent results
            if self._cache_manager and len(classifications) == 1:
                target_agent, condensed, confidence = classifications[0]
                if target_agent != _FALLBACK_AGENT:
                    try:
                        self._cache_manager.store_routing(user_text, target_agent, confidence, condensed)
                    except Exception:
                        logger.warning("Failed to store routing decision", exc_info=True)
            return classifications, False
        except Exception:
            logger.exception("Intent classification failed, falling back to %s", _FALLBACK_AGENT)
            return [(_FALLBACK_AGENT, user_text, 0.0)], False

    async def _parse_classification(self, response: str, original_text: str) -> list[tuple[str, str, float]]:
        """Parse LLM classification response (single or multi-line).

        Expected format per line: "<agent-id> (<confidence>%): <condensed task>"
        Falls back to old format: "<agent-id>: <condensed task>"
        Falls back to general-agent if parsing fails.

        Returns a list of (agent_id, condensed_task, confidence) tuples.
        """
        response = response.strip()
        known_agents = await self._get_known_agents()
        results: list[tuple[str, str, float]] = []

        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines:
            # Strip [SEQ] prefix used for sequential send dispatch
            if line.startswith("[SEQ]"):
                line = line[5:].strip()
            # Try new format: "agent-id (85%): task text"
            match = re.match(r'^([\w-]+)\s*\((\d+)%?\)\s*:\s*(.+)$', line, re.DOTALL)
            if match:
                agent_id = match.group(1).strip().lower()
                confidence = min(float(match.group(2)) / 100.0, 1.0)
                condensed = match.group(3).strip()
            else:
                # Fallback to old format: "agent-id: task text"
                if ":" not in line:
                    logger.warning("Could not parse classification line: %s", line[:100])
                    continue
                agent_id, _, condensed = line.partition(":")
                agent_id = agent_id.strip().lower()
                condensed = condensed.strip()
                confidence = 0.8

            if agent_id not in known_agents:
                logger.warning("Unknown agent '%s' in classification, skipping line", agent_id)
                continue

            if not condensed:
                condensed = original_text

            results.append((agent_id, condensed, confidence))

        if not results:
            return [(_FALLBACK_AGENT, original_text, 0.0)]

        # Deduplicate by agent_id: keep higher confidence, merge tasks
        seen: dict[str, tuple[str, float, list[str]]] = {}
        for agent_id, condensed, confidence in results:
            if agent_id in seen:
                existing_condensed, existing_conf, tasks = seen[agent_id]
                if confidence > existing_conf:
                    tasks.append(existing_condensed)
                    seen[agent_id] = (condensed, confidence, tasks)
                else:
                    tasks.append(condensed)
            else:
                seen[agent_id] = (condensed, confidence, [])

        deduped = []
        for agent_id, (condensed, confidence, extra_tasks) in seen.items():
            if extra_tasks:
                condensed = condensed + " ; " + " ; ".join(extra_tasks)
            deduped.append((agent_id, condensed, confidence))

        # Sort by confidence desc, cap at 3
        deduped.sort(key=lambda x: x[2], reverse=True)
        return deduped[:3]

    def _get_turns(self, conversation_id: str | None) -> list[dict]:
        """Get recent conversation turns for context."""
        if not conversation_id:
            return []
        entry = self._conversations.get(conversation_id)
        if entry is None:
            return []
        ts, turns = entry
        if time.monotonic() - ts > _CONVERSATION_TTL_SECONDS:
            self._conversations.pop(conversation_id, None)
            return []
        return list(turns)

    def _store_turn(self, conversation_id: str | None, user_text: str, assistant_text: str, agent_id: str | None = None) -> None:
        """Store a conversation turn, keeping last _MAX_TURNS exchanges."""
        if not conversation_id:
            return
        self._evict_stale_conversations()
        now = time.monotonic()
        if conversation_id in self._conversations:
            self._conversations.move_to_end(conversation_id)
            _, turns = self._conversations[conversation_id]
        else:
            turns = []
        turns.append({"role": "user", "content": user_text})
        assistant_turn = {"role": "assistant", "content": assistant_text}
        if agent_id:
            assistant_turn["agent_id"] = agent_id
        turns.append(assistant_turn)
        max_messages = _MAX_TURNS * 2
        if len(turns) > max_messages:
            turns = turns[-max_messages:]
        self._conversations[conversation_id] = (now, turns)

    def _evict_stale_conversations(self) -> None:
        """Remove conversations older than TTL and enforce max count."""
        now = time.monotonic()
        while self._conversations:
            oldest_key = next(iter(self._conversations))
            ts, _ = self._conversations[oldest_key]
            if now - ts > _CONVERSATION_TTL_SECONDS:
                self._conversations.pop(oldest_key)
            else:
                break
        while len(self._conversations) > _MAX_CONVERSATIONS:
            self._conversations.popitem(last=False)

    async def _merge_responses(
        self,
        agent_responses: list[tuple[str, str, bool]],
        user_text: str,
    ) -> str:
        """Merge multiple agent responses into a single natural answer via LLM.

        Always calls LLM regardless of personality settings.
        Includes personality prompt if configured.
        Falls back to bracket-prefixed format on failure.
        """
        if not agent_responses:
            return "I couldn't process that request."

        # Only one response: return it directly
        if len(agent_responses) == 1:
            return agent_responses[0][1] or "I couldn't process that request."

        # Build structured summary of each agent response
        summary_parts = []
        for agent_id, speech, acted in agent_responses:
            status = "[action executed]" if acted else "[no action executed]"
            if speech and speech.strip():
                summary_parts.append(f"- {agent_id} {status}: {speech}")
            else:
                summary_parts.append(f"- {agent_id} {status}: (no response)")
        agent_summary = "\n".join(summary_parts)

        try:
            personality = ""
            try:
                personality = await SettingsRepository.get_value("personality.prompt", "")
            except Exception:
                pass

            system_content = self._load_prompt("merge")
            personality_text = personality.strip() if personality and personality.strip() else ""
            system_content = system_content.replace("{personality}", personality_text).strip()

            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"User asked: {user_text}\n\n"
                        f"Agent responses:\n{agent_summary}\n\n"
                        "Combine into one natural response:"
                    ),
                },
            ]

            overrides = {
                "temperature": self._mediation_temperature,
                "max_tokens": self._mediation_max_tokens,
            }
            if self._mediation_model:
                overrides["model"] = self._mediation_model
            result = await self._call_llm(messages, **overrides)
            return result.strip() if result and result.strip() else self._format_fallback(agent_responses)
        except Exception:
            logger.warning("Multi-agent response merge failed, using fallback format", exc_info=True)
            return self._format_fallback(agent_responses)

    @staticmethod
    def _format_fallback(agent_responses: list[tuple[str, str, bool]]) -> str:
        """Fallback formatting when LLM merge fails."""
        parts = [f"[{aid}] {sp}" for aid, sp, _ in agent_responses if sp and sp.strip()]
        return "\n\n".join(parts) if parts else "I couldn't process that request."

    async def _mediate_response(self, agent_speech: str, user_text: str, agent_id: str, language: str = "en") -> str:
        """Optionally mediate the domain agent response with personality.

        When personality.prompt is non-empty, passes the agent speech through
        a lightweight LLM call to apply the configured personality.
        Falls back to the original speech on any failure.
        """
        try:
            personality = await SettingsRepository.get_value("personality.prompt", "")
            if not personality.strip():
                return agent_speech
        except Exception:
            return agent_speech

        if not agent_speech or not agent_speech.strip():
            return agent_speech

        try:
            system_prompt = self._load_prompt("mediate")
            personality_text = personality.strip() if personality.strip() else ""
            system_prompt = system_prompt.replace("{personality}", personality_text)
            system_prompt = system_prompt.replace("{language}", language or "en").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"User asked: {user_text}\nAgent ({agent_id}) responded: {agent_speech}\n\nRephrase in {language}:",
                },
            ]
            overrides = {
                "temperature": self._mediation_temperature,
                "max_tokens": self._mediation_max_tokens,
            }
            if self._mediation_model:
                overrides["model"] = self._mediation_model
            result = await self._call_llm(messages, **overrides)
            return result.strip() if result and result.strip() else agent_speech
        except Exception:
            logger.warning("Response mediation failed, using original speech", exc_info=True)
            return agent_speech

    @staticmethod
    def _strip_seq_rule(prompt: str) -> str:
        """Remove the sequential dispatch rule block when send-agent is unavailable."""
        start_marker = "Sequential dispatch rule:"
        end_marker = "Format:"
        start_idx = prompt.find(start_marker)
        end_idx = prompt.find(end_marker)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return prompt[:start_idx] + prompt[end_idx:]
        return prompt
