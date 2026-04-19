"""Shared verification tests for domain executors (0.18.5 FLOW-VERIFY-SHARED).

Covers:
- ``call_service_with_verification`` primitive in ``action_executor``.
- Each domain executor (climate, media, music, security, scene,
  automation, timer) using it instead of the legacy
  ``asyncio.sleep(0.3) + get_state`` dance.

The scenario we are pinning down is the async-bus actor case: HA's REST
``call_service`` returns ``[]`` immediately but the state change fires via
WebSocket slightly later. Without the shared helper we'd speak the stale
state; with it we must report the verified target state.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

_litellm_mock = MagicMock()


class _AuthenticationError(Exception):
    pass


_litellm_mock.exceptions.AuthenticationError = _AuthenticationError
sys.modules.setdefault("litellm", _litellm_mock)


from app.agents.action_executor import (  # noqa: E402
    build_verified_speech,
    call_service_with_verification,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeMatch:
    """Minimal stand-in for ``EntityMatcher.MatchResult``."""

    entity_id: str
    friendly_name: str
    score: float = 0.95
    signal_scores: dict[str, float] = field(default_factory=dict)


def _make_matcher(entity_id: str, friendly_name: str):
    """An entity_matcher whose ``.match`` returns a single fixed match."""
    matcher = MagicMock()
    matcher.match = AsyncMock(return_value=[_FakeMatch(entity_id, friendly_name)])
    return matcher


def _attach_ws_observer_shim(client, *, observed_state: str | None):
    """Install ``expect_state`` that yields an observer pre-populated with
    ``observed_state``.

    Simulates the async-bus case: WS fires *during* the ``with`` block, so
    when the caller exits the context ``observer["new_state"]`` is already
    set. ``get_state`` is intentionally broken so the test fails loudly
    if anyone falls back to it.
    """

    @asynccontextmanager
    async def _expect_state(
        entity_id,
        *,
        expected=None,
        timeout=0.05,
        poll_interval=0.01,
        poll_max=0.05,
    ):
        observer = {"new_state": observed_state}
        yield observer

    client.expect_state = _expect_state
    client.set_state_observer = MagicMock()
    client.get_state = AsyncMock(
        side_effect=AssertionError(
            "domain executors must not call get_state directly anymore - use call_service_with_verification",
        ),
    )
    return client


def _make_ha_client(*, call_result=None, observed_state: str | None = None):
    client = MagicMock()
    client.call_service = AsyncMock(return_value=call_result)
    _attach_ws_observer_shim(client, observed_state=observed_state)
    return client


# ---------------------------------------------------------------------------
# call_service_with_verification primitive
# ---------------------------------------------------------------------------


class TestCallServiceWithVerification:
    """Direct unit tests for the shared helper."""

    @pytest.mark.asyncio
    async def test_non_empty_rest_response_is_authoritative(self):
        client = _make_ha_client(
            call_result=[{"entity_id": "light.x", "state": "on"}],
            observed_state="off",  # stale, must be ignored
        )
        result = await call_service_with_verification(
            client,
            "light",
            "turn_on",
            "light.x",
            expected_state="on",
        )
        assert result["success"] is True
        assert result["observed_state"] == "on"
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_empty_rest_falls_back_to_ws_observer(self):
        client = _make_ha_client(call_result=[], observed_state="on")
        result = await call_service_with_verification(
            client,
            "light",
            "turn_on",
            "light.x",
            expected_state="on",
        )
        assert result["success"] is True
        assert result["observed_state"] == "on"
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_empty_rest_no_observer_returns_unverified(self):
        client = _make_ha_client(call_result=[], observed_state=None)
        result = await call_service_with_verification(
            client,
            "light",
            "turn_on",
            "light.x",
            expected_state="on",
        )
        assert result["success"] is True
        assert result["observed_state"] is None
        assert result["verified"] is False

    @pytest.mark.asyncio
    async def test_call_service_exception_surfaces_failure(self):
        client = _make_ha_client(observed_state="on")
        client.call_service = AsyncMock(side_effect=RuntimeError("boom"))
        result = await call_service_with_verification(
            client,
            "light",
            "turn_on",
            "light.x",
            expected_state="on",
        )
        assert result["success"] is False
        assert result["verified"] is False
        assert isinstance(result["error"], RuntimeError)

    @pytest.mark.asyncio
    async def test_no_expected_accepts_any_observed_change(self):
        client = _make_ha_client(call_result=[], observed_state="playing")
        result = await call_service_with_verification(
            client,
            "media_player",
            "media_play",
            "media_player.x",
            expected_state=None,
        )
        assert result["verified"] is True
        assert result["observed_state"] == "playing"


# ---------------------------------------------------------------------------
# build_verified_speech
# ---------------------------------------------------------------------------


class TestBuildVerifiedSpeech:
    def test_verified_uses_expected_state(self):
        speech = build_verified_speech(
            friendly_name="Front Door",
            action_name="lock",
            expected_state="locked",
            observed_state="locked",
            verified=True,
            action_phrases={"lock": "locked"},
        )
        assert speech == "Done, Front Door is now locked."

    def test_unverified_with_expected_falls_back_to_intent(self):
        speech = build_verified_speech(
            friendly_name="Front Door",
            action_name="lock",
            expected_state="locked",
            observed_state=None,
            verified=False,
            action_phrases={"lock": "locked"},
        )
        # Intent-first phrasing takes precedence over the expected-state
        # fallback when an action phrase is registered.
        assert speech == "Done, Front Door locked."

    def test_stale_observation_does_not_override_expected(self):
        speech = build_verified_speech(
            friendly_name="Keller",
            action_name="turn_off",
            expected_state="off",
            observed_state="on",
            verified=False,
            action_phrases={"turn_off": "turned off"},
        )
        assert "is now on" not in speech
        assert speech == "Done, Keller turned off."

    def test_falls_back_to_humanized_action_name(self):
        speech = build_verified_speech(
            friendly_name="Thermostat",
            action_name="set_fan_mode",
            expected_state=None,
            observed_state=None,
            verified=False,
            action_phrases=None,
        )
        assert speech == "Done, Thermostat set fan mode."


# ---------------------------------------------------------------------------
# Per-executor WS-observer success tests
# ---------------------------------------------------------------------------

# The helper below drives each domain executor through the "REST empty +
# WS observer confirms" path. It proves:
#   * the executor uses the shared verification helper (``get_state`` is
#     wired to raise if called),
#   * the verified state is reported in ``new_state`` and speech,
#   * the service call was awaited with the expected domain/service.


async def _assert_verified_action(
    executor,
    *,
    action: dict,
    entity_id: str,
    friendly_name: str,
    observed_state: str | None,
    expected_domain: str,
    expected_service: str,
    expected_new_state: str | None,
    speech_assertions: list[str],
    negative_speech_assertions: list[str] | None = None,
    call_result=None,
):
    ha_client = _make_ha_client(
        call_result=[] if call_result is None else call_result,
        observed_state=observed_state,
    )
    matcher = _make_matcher(entity_id, friendly_name)
    entity_index = MagicMock()

    result = await executor(action, ha_client, entity_index, matcher)

    assert result["success"] is True, result
    assert result["entity_id"] == entity_id
    assert result["new_state"] == expected_new_state
    for needle in speech_assertions:
        assert needle in result["speech"], result["speech"]
    for needle in negative_speech_assertions or []:
        assert needle not in result["speech"], result["speech"]
    ha_client.call_service.assert_awaited_once()
    call_args = ha_client.call_service.await_args
    assert call_args.args[0] == expected_domain
    assert call_args.args[1] == expected_service


# ---- climate ---------------------------------------------------------------


class TestClimateExecutorVerification:
    @pytest.mark.asyncio
    async def test_turn_off_empty_rest_ws_confirms(self):
        from app.agents.climate_executor import execute_climate_action

        await _assert_verified_action(
            execute_climate_action,
            action={"action": "turn_off", "entity": "living room"},
            entity_id="climate.living_room",
            friendly_name="Living Room",
            observed_state="off",
            expected_domain="climate",
            expected_service="turn_off",
            expected_new_state="off",
            speech_assertions=["Living Room", "off"],
        )

    @pytest.mark.asyncio
    async def test_set_hvac_mode_uses_dynamic_expected(self):
        from app.agents.climate_executor import execute_climate_action

        await _assert_verified_action(
            execute_climate_action,
            action={
                "action": "set_hvac_mode",
                "entity": "living room",
                "parameters": {"hvac_mode": "heat"},
            },
            entity_id="climate.living_room",
            friendly_name="Living Room",
            observed_state="heat",
            expected_domain="climate",
            expected_service="set_hvac_mode",
            expected_new_state="heat",
            speech_assertions=["Living Room", "heat"],
        )

    @pytest.mark.asyncio
    async def test_set_hvac_mode_stale_observation_falls_back_to_intent(self):
        """Observer saw the *old* mode -- don't contradict intent."""
        from app.agents.climate_executor import execute_climate_action

        ha_client = _make_ha_client(
            call_result=[],
            observed_state="cool",  # stale: we asked for heat
        )
        matcher = _make_matcher("climate.living_room", "Living Room")
        result = await execute_climate_action(
            {
                "action": "set_hvac_mode",
                "entity": "living room",
                "parameters": {"hvac_mode": "heat"},
            },
            ha_client,
            MagicMock(),
            matcher,
        )
        assert result["success"] is True
        assert "is now cool" not in result["speech"]


# ---- media -----------------------------------------------------------------


class TestMediaExecutorVerification:
    @pytest.mark.asyncio
    async def test_play_empty_rest_ws_confirms_playing(self):
        from app.agents.media_executor import execute_media_action

        await _assert_verified_action(
            execute_media_action,
            action={"action": "play", "entity": "living room tv"},
            entity_id="media_player.living_room_tv",
            friendly_name="Living Room TV",
            observed_state="playing",
            expected_domain="media_player",
            expected_service="media_play",
            expected_new_state="playing",
            speech_assertions=["Living Room TV", "playing"],
        )

    @pytest.mark.asyncio
    async def test_turn_off_stale_observed_on_does_not_speak_on(self):
        from app.agents.media_executor import execute_media_action

        ha_client = _make_ha_client(call_result=[], observed_state="playing")
        matcher = _make_matcher("media_player.tv", "TV")
        result = await execute_media_action(
            {"action": "turn_off", "entity": "tv"},
            ha_client,
            MagicMock(),
            matcher,
        )
        assert result["success"] is True
        assert "is now playing" not in result["speech"]
        assert "TV" in result["speech"]


# ---- music -----------------------------------------------------------------


class TestMusicExecutorVerification:
    @pytest.mark.asyncio
    async def test_media_pause_empty_rest_ws_confirms_paused(self):
        from app.agents.music_executor import execute_music_action

        await _assert_verified_action(
            execute_music_action,
            action={"action": "media_pause", "entity": "kitchen speaker"},
            entity_id="media_player.kitchen_speaker",
            friendly_name="Kitchen Speaker",
            observed_state="paused",
            expected_domain="media_player",
            expected_service="media_pause",
            expected_new_state="paused",
            speech_assertions=["Kitchen Speaker", "paused"],
        )


# ---- security --------------------------------------------------------------


class TestSecurityExecutorVerification:
    @pytest.mark.asyncio
    async def test_lock_empty_rest_ws_confirms_locked(self):
        from app.agents.security_executor import execute_security_action

        await _assert_verified_action(
            execute_security_action,
            action={"action": "lock", "entity": "front door"},
            entity_id="lock.front_door",
            friendly_name="Front Door",
            observed_state="locked",
            expected_domain="lock",
            expected_service="lock",
            expected_new_state="locked",
            speech_assertions=["Front Door", "locked"],
        )

    @pytest.mark.asyncio
    async def test_alarm_arm_home_stale_disarmed_does_not_report_disarmed(self):
        """Critical safety test: never claim the alarm is disarmed when we
        issued an arm command."""
        from app.agents.security_executor import execute_security_action

        ha_client = _make_ha_client(call_result=[], observed_state="disarmed")
        matcher = _make_matcher("alarm_control_panel.home", "Alarm")
        result = await execute_security_action(
            {"action": "alarm_arm_home", "entity": "alarm"},
            ha_client,
            MagicMock(),
            matcher,
        )
        assert result["success"] is True
        assert "disarmed" not in result["speech"]
        assert "armed" in result["speech"]


# ---- scene -----------------------------------------------------------------


class TestSceneExecutorVerification:
    @pytest.mark.asyncio
    async def test_activate_scene_fires_and_reports_activation(self):
        from app.agents.scene_executor import execute_scene_action

        await _assert_verified_action(
            execute_scene_action,
            action={"action": "activate_scene", "entity": "movie night"},
            entity_id="scene.movie_night",
            friendly_name="Movie Night",
            observed_state="2026-04-19T12:00:00Z",  # scene timestamp
            expected_domain="scene",
            expected_service="turn_on",
            expected_new_state="2026-04-19T12:00:00Z",
            speech_assertions=["Movie Night", "activated"],
        )


# ---- automation ------------------------------------------------------------


class TestAutomationExecutorVerification:
    @pytest.mark.asyncio
    async def test_enable_automation_ws_confirms_on(self):
        from app.agents.automation_executor import execute_automation_action

        await _assert_verified_action(
            execute_automation_action,
            action={"action": "enable_automation", "entity": "morning lights"},
            entity_id="automation.morning_lights",
            friendly_name="Morning Lights",
            observed_state="on",
            expected_domain="automation",
            expected_service="turn_on",
            expected_new_state="on",
            speech_assertions=["Morning Lights"],
        )

    @pytest.mark.asyncio
    async def test_trigger_automation_no_expected_state_uses_intent_phrase(self):
        """Triggering does NOT change the entity state; speech must be
        intent-first ("triggered"), not "is now on"."""
        from app.agents.automation_executor import execute_automation_action

        ha_client = _make_ha_client(call_result=[], observed_state="on")
        matcher = _make_matcher("automation.morning_lights", "Morning Lights")
        result = await execute_automation_action(
            {"action": "trigger_automation", "entity": "morning lights"},
            ha_client,
            MagicMock(),
            matcher,
        )
        assert result["success"] is True
        assert "triggered" in result["speech"]


# ---- timer -----------------------------------------------------------------


class TestTimerExecutorVerification:
    @pytest.mark.asyncio
    async def test_start_timer_ws_confirms_active(self):
        from app.agents.timer_executor import _timer_pool, execute_timer_action

        _timer_pool._name_to_entity.clear()
        _timer_pool._entity_to_name.clear()
        _timer_pool._entity_to_metadata.clear()

        ha_client = _make_ha_client(call_result=[], observed_state="active")
        matcher = _make_matcher("timer.pasta", "Pasta")
        result = await execute_timer_action(
            {
                "action": "start_timer",
                "entity": "pasta",
                "parameters": {"duration": "00:05:00"},
            },
            ha_client,
            MagicMock(),
            matcher,
        )
        assert result["success"] is True
        assert result["new_state"] == "active"
        assert "Pasta" in result["speech"]
        assert "active" in result["speech"] or "started" in result["speech"]

    @pytest.mark.asyncio
    async def test_cancel_timer_stale_active_does_not_speak_active(self):
        from app.agents.timer_executor import execute_timer_action

        ha_client = _make_ha_client(call_result=[], observed_state="active")
        matcher = _make_matcher("timer.pasta", "Pasta")
        result = await execute_timer_action(
            {"action": "cancel_timer", "entity": "pasta"},
            ha_client,
            MagicMock(),
            matcher,
        )
        assert result["success"] is True
        assert "is now active" not in result["speech"]
