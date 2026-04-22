"""Tests for cached-action visibility re-check, empty-result handling, and
sequential-send canned-string filtering (FLOW-CRIT-1, FLOW-CRIT-2, FLOW-CRIT-3)."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# litellm is not installed in the test environment; provide a stub so
# importing the orchestrator module does not fail.
_litellm_mock = MagicMock()


class _AuthenticationError(Exception):
    pass


_litellm_mock.exceptions.AuthenticationError = _AuthenticationError
sys.modules.setdefault("litellm", _litellm_mock)

from app.agents.orchestrator import OrchestratorAgent  # noqa: E402
from app.cache.cache_manager import CacheResult  # noqa: E402
from tests.helpers import make_cached_action, make_response_cache_entry  # noqa: E402

pytestmark = pytest.mark.asyncio


def _make_orchestrator():
    dispatcher = AsyncMock()
    cache_manager = MagicMock()
    cache_manager.apply_rewrite = AsyncMock()
    cache_manager.invalidate_response = MagicMock()
    ha_client = AsyncMock()
    # ``call_service_with_verification`` calls ``expect_state(...)`` when present;
    # a bare AsyncMock returns an un-awaited coroutine. Use REST-only path in tests.
    ha_client.expect_state = None
    orch = OrchestratorAgent(
        dispatcher=dispatcher,
        cache_manager=cache_manager,
        ha_client=ha_client,
    )
    return orch, dispatcher, cache_manager, ha_client


def _make_cache_result(
    *,
    cached_action,
    agent_id: str = "light-agent",
    query_text: str = "turn on the kitchen light",
):
    entry = make_response_cache_entry(
        query_text=query_text,
        agent_id=agent_id,
        cached_action=cached_action,
    )
    return CacheResult(
        hit_type="response_hit",
        agent_id=agent_id,
        response_text=entry.response_text,
        cached_action=cached_action,
        entry=entry,
        similarity=0.99,
    )


# ---------------------------------------------------------------------------
# FLOW-CRIT-1: visibility re-check on cached-action replay
# ---------------------------------------------------------------------------


class TestCachedActionVisibility:
    async def test_cached_action_blocked_when_entity_revoked(self):
        """Visibility was revoked after caching: must invalidate + fall through."""
        orch, _dispatcher, cache_manager, ha_client = _make_orchestrator()
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")
        cache_result = _make_cache_result(cached_action=cached)

        # Visibility rules now exclude the light domain entirely for this agent.
        rules = [{"rule_type": "domain_include", "rule_value": "switch"}]
        with patch(
            "app.db.repository.EntityVisibilityRepository.get_rules",
            new=AsyncMock(return_value=rules),
        ):
            result = await orch._handle_response_cache_hit(
                cache_result,
                "conv-1",
                "turn on the kitchen light",
                None,
            )

        assert result is None, "Cache hit must fall through when entity not visible"
        ha_client.call_service.assert_not_called()
        cache_manager.invalidate_response.assert_called_once()

    async def test_cached_action_executes_when_entity_still_visible(self):
        """Visibility unchanged: cached action runs normally."""
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        ha_client.call_service.return_value = [{"entity_id": "light.kitchen", "state": "on"}]
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")
        cache_result = _make_cache_result(cached_action=cached)

        rules = [{"rule_type": "domain_include", "rule_value": "light"}]
        with (
            patch(
                "app.db.repository.EntityVisibilityRepository.get_rules",
                new=AsyncMock(return_value=rules),
            ),
            patch(
                "app.agents.orchestrator.ConversationRepository.insert",
                new=AsyncMock(return_value=1),
            ),
        ):
            result = await orch._handle_response_cache_hit(
                cache_result,
                "conv-1",
                "turn on the kitchen light",
                None,
            )

        ha_client.call_service.assert_called_once()
        assert result is not None
        assert result.get("speech")

    async def test_cached_action_no_rules_means_full_access(self):
        """No visibility rules at all: cached action executes (matches live matcher)."""
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        ha_client.call_service.return_value = [{"entity_id": "light.kitchen", "state": "on"}]
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")
        cache_result = _make_cache_result(cached_action=cached)

        with (
            patch(
                "app.db.repository.EntityVisibilityRepository.get_rules",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.agents.orchestrator.ConversationRepository.insert",
                new=AsyncMock(return_value=1),
            ),
        ):
            result = await orch._handle_response_cache_hit(
                cache_result,
                "conv-1",
                "turn on the kitchen light",
                None,
            )

        ha_client.call_service.assert_called_once()
        assert result is not None

    async def test_visibility_lookup_failure_fails_closed(self):
        """If visibility cannot be evaluated, treat as not visible."""
        orch, _dispatcher, cache_manager, ha_client = _make_orchestrator()
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")
        cache_result = _make_cache_result(cached_action=cached)

        with patch(
            "app.db.repository.EntityVisibilityRepository.get_rules",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            result = await orch._handle_response_cache_hit(
                cache_result,
                "conv-1",
                "turn on the kitchen light",
                None,
            )

        assert result is None
        ha_client.call_service.assert_not_called()
        cache_manager.invalidate_response.assert_called_once()


# ---------------------------------------------------------------------------
# FLOW-CRIT-2: empty HA REST response treated as cache miss
# ---------------------------------------------------------------------------


class TestCachedActionEmptyResponse:
    async def test_empty_list_response_returns_none(self):
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        ha_client.call_service.return_value = []
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")

        result = await orch._execute_cached_action(cached)

        assert result is None
        ha_client.call_service.assert_called_once()

    async def test_empty_dict_response_returns_none(self):
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        ha_client.call_service.return_value = {}
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")

        result = await orch._execute_cached_action(cached)

        assert result is None

    async def test_non_empty_list_response_passes_through(self):
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        payload = [{"entity_id": "light.kitchen", "state": "on"}]
        ha_client.call_service.return_value = payload
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")

        result = await orch._execute_cached_action(cached)

        # ``_execute_cached_action`` wraps the REST reply into a
        # result dict so the rest of the cache-replay pipeline can
        # reason about ``success``/``state``/``source`` uniformly.
        assert result is not None
        assert result["success"] is True
        assert result["entity_id"] == "light.kitchen"
        assert result["action"] == "turn_on"
        assert result["state"] == "on"
        assert result["source"] == "call_service"

    async def test_cached_action_empty_ha_response_falls_through(self):
        """End-to-end: empty HA result causes _handle_response_cache_hit
        to return None so the orchestrator falls through to live dispatch."""
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        ha_client.call_service.return_value = []
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")
        cache_result = _make_cache_result(cached_action=cached)

        # Visibility allows the entity so we exercise the empty-result path.
        rules = [{"rule_type": "domain_include", "rule_value": "light"}]
        with patch(
            "app.db.repository.EntityVisibilityRepository.get_rules",
            new=AsyncMock(return_value=rules),
        ):
            result = await orch._handle_response_cache_hit(
                cache_result,
                "conv-1",
                "turn on the kitchen light",
                None,
            )

        assert result is None
        ha_client.call_service.assert_called_once()


# ---------------------------------------------------------------------------
# FLOW-CRIT-3: sequential-send must not pipe canned content-failure text
# ---------------------------------------------------------------------------


class TestSequentialSendContentFailure:
    def _orchestrator_for_send(self):
        orch, dispatcher, _cache_manager, _ha_client = _make_orchestrator()
        # ha_client must be falsy in _dispatch_single's home-context branch
        # to avoid network / zoneinfo dependencies during this unit test.
        orch._ha_client = None
        return orch, dispatcher

    async def test_skips_send_when_content_dispatch_times_out(self):
        """Content agent timed out -> _dispatch_single returns canned text +
        result=None. Send-agent must NOT be invoked."""
        orch, _dispatcher = self._orchestrator_for_send()

        async def fake_dispatch_single(*args, **kwargs):
            # Mimic the timeout fallback shape from _dispatch_single.
            return ("general-agent", "I couldn't process that request in time.", None)

        with patch.object(orch, "_dispatch_single", side_effect=fake_dispatch_single) as mock_ds:
            classifications = [
                ("general-agent", "summarize today", 0.9),
                ("send-agent", "telegram", 0.9),
            ]
            routed_to, speech, result = await orch._handle_sequential_send(
                classifications,
                user_text="send today summary to telegram",
                conversation_id="conv-1",
                turns=[],
                span_collector=None,
                incoming_context=None,
            )

        # Only the content dispatch should have happened; no send dispatch.
        assert mock_ds.call_count == 1
        assert routed_to == "send-agent"
        assert speech == "I could not prepare the content to send."
        assert result and result.get("error", {}).get("code") == "content_unavailable"

    async def test_skips_send_when_content_returns_error(self):
        """Content agent returned a result dict with `error` -- skip send."""
        orch, _dispatcher = self._orchestrator_for_send()

        async def fake_dispatch_single(*args, **kwargs):
            return (
                "general-agent",
                "Sorry, that didn't work.",
                {
                    "speech": "Sorry, that didn't work.",
                    "error": {
                        "code": "llm_error",
                        "recoverable": True,
                    },
                },
            )

        with patch.object(orch, "_dispatch_single", side_effect=fake_dispatch_single) as mock_ds:
            classifications = [
                ("general-agent", "summarize today", 0.9),
                ("send-agent", "telegram", 0.9),
            ]
            _routed_to, speech, result = await orch._handle_sequential_send(
                classifications,
                user_text="send today summary to telegram",
                conversation_id="conv-1",
                turns=[],
                span_collector=None,
                incoming_context=None,
            )

        assert mock_ds.call_count == 1
        assert speech == "I could not prepare the content to send."
        assert result["error"]["code"] == "content_unavailable"

    async def test_skips_send_when_content_returns_partial_failure(self):
        orch, _dispatcher = self._orchestrator_for_send()

        async def fake_dispatch_single(*args, **kwargs):
            return (
                "general-agent",
                "Partial result text.",
                {"speech": "Partial result text.", "partial_failure": True},
            )

        with patch.object(orch, "_dispatch_single", side_effect=fake_dispatch_single) as mock_ds:
            classifications = [
                ("general-agent", "summarize", 0.9),
                ("send-agent", "telegram", 0.9),
            ]
            _routed, speech, result = await orch._handle_sequential_send(
                classifications,
                user_text="x",
                conversation_id="conv-1",
                turns=[],
                span_collector=None,
                incoming_context=None,
            )

        assert mock_ds.call_count == 1
        assert speech == "I could not prepare the content to send."
        assert result["error"]["code"] == "content_unavailable"

    async def test_send_proceeds_on_successful_content(self):
        """Content succeeded -> send-agent dispatch happens normally."""
        orch, _dispatcher = self._orchestrator_for_send()

        call_log: list[str] = []

        async def fake_dispatch_single(target_agent, *args, **kwargs):
            call_log.append(target_agent)
            if target_agent == "send-agent":
                return (
                    "send-agent",
                    "Sent.",
                    {"speech": "Sent."},
                )
            return (
                target_agent,
                "Today: light usage normal.",
                {"speech": "Today: light usage normal."},
            )

        with patch.object(orch, "_dispatch_single", side_effect=fake_dispatch_single):
            classifications = [
                ("general-agent", "summarize today", 0.9),
                ("send-agent", "telegram", 0.9),
            ]
            _routed, speech, result = await orch._handle_sequential_send(
                classifications,
                user_text="send today summary to telegram",
                conversation_id="conv-1",
                turns=[],
                span_collector=None,
                incoming_context=None,
            )

        assert call_log == ["general-agent", "send-agent"]
        assert speech == "Sent."
        assert (result or {}).get("error") is None


class TestActionCacheTraceDualWrite:
    async def test_orchestrator_writes_both_action_and_legacy_metadata_keys(self):
        orch, _dispatcher, _cache_manager, ha_client = _make_orchestrator()
        ha_client.call_service.return_value = [{"entity_id": "light.kitchen", "state": "on"}]
        cached = make_cached_action(service="light/turn_on", entity_id="light.kitchen")
        cache_result = _make_cache_result(cached_action=cached)

        from app.analytics.tracer import SpanCollector

        span_collector = SpanCollector("dual-write-test")
        with (
            patch(
                "app.db.repository.EntityVisibilityRepository.get_rules",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.agents.orchestrator.ConversationRepository.insert",
                new=AsyncMock(return_value=1),
            ),
        ):
            result = await orch._handle_response_cache_hit(
                cache_result,
                "conv-1",
                "turn on the kitchen light",
                span_collector,
            )

        assert result is not None
        return_spans = [s for s in span_collector._spans if s.get("span_name") == "return"]
        assert return_spans, "orchestrator must emit a 'return' span on cache hit"
        meta = return_spans[-1].get("metadata") or {}
        assert meta.get("action_cache_hit") is True
        assert meta.get("response_cache_hit") is True
