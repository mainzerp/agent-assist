"""Tests for app.presence -- sensors, scoring, and detector."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.presence.sensors import discover_sensors, PresenceSensor, MOTION_CLASSES, OCCUPANCY_CLASSES
from app.presence.scoring import compute_room_confidence, SensorEvent, SENSOR_WEIGHTS
from app.presence.detector import PresenceDetector


# ---------------------------------------------------------------------------
# Sensor discovery
# ---------------------------------------------------------------------------

class TestDiscoverSensors:

    async def test_discovers_motion_sensor(self):
        ha = AsyncMock()
        ha.get_states = AsyncMock(return_value=[
            {
                "entity_id": "binary_sensor.hallway_motion",
                "state": "on",
                "attributes": {
                    "device_class": "motion",
                    "friendly_name": "Hallway Motion",
                    "area_id": "hallway",
                },
            },
        ])
        sensors = await discover_sensors(ha)
        assert len(sensors) == 1
        assert sensors[0].sensor_type == "motion"
        assert sensors[0].area == "hallway"

    async def test_discovers_occupancy_sensor(self):
        ha = AsyncMock()
        ha.get_states = AsyncMock(return_value=[
            {
                "entity_id": "binary_sensor.office_occupancy",
                "state": "off",
                "attributes": {
                    "device_class": "occupancy",
                    "friendly_name": "Office Occupancy",
                    "area_id": "office",
                },
            },
        ])
        sensors = await discover_sensors(ha)
        assert len(sensors) == 1
        assert sensors[0].sensor_type == "occupancy"

    async def test_discovers_mmwave_by_name(self):
        ha = AsyncMock()
        ha.get_states = AsyncMock(return_value=[
            {
                "entity_id": "binary_sensor.bedroom_mmwave",
                "state": "on",
                "attributes": {
                    "device_class": "",
                    "friendly_name": "Bedroom mmWave Sensor",
                    "area_id": "bedroom",
                },
            },
        ])
        sensors = await discover_sensors(ha)
        assert len(sensors) == 1
        assert sensors[0].sensor_type == "mmwave"

    async def test_ignores_non_binary_sensor_entities(self):
        ha = AsyncMock()
        ha.get_states = AsyncMock(return_value=[
            {
                "entity_id": "light.kitchen",
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Light"},
            },
        ])
        sensors = await discover_sensors(ha)
        assert len(sensors) == 0

    async def test_ignores_irrelevant_binary_sensors(self):
        ha = AsyncMock()
        ha.get_states = AsyncMock(return_value=[
            {
                "entity_id": "binary_sensor.door_contact",
                "state": "off",
                "attributes": {
                    "device_class": "door",
                    "friendly_name": "Front Door",
                },
            },
        ])
        sensors = await discover_sensors(ha)
        assert len(sensors) == 0

    async def test_returns_empty_on_no_states(self):
        ha = AsyncMock()
        ha.get_states = AsyncMock(return_value=[])
        sensors = await discover_sensors(ha)
        assert sensors == []


# ---------------------------------------------------------------------------
# Room scoring
# ---------------------------------------------------------------------------

class TestComputeRoomConfidence:

    def test_recent_mmwave_event_scores_high(self):
        now = time.time()
        events = [SensorEvent(sensor_type="mmwave", area="kitchen", triggered_at=now)]
        scores = compute_room_confidence(events)
        assert scores["kitchen"] == pytest.approx(1.0, abs=0.01)

    def test_motion_event_scores_lower_than_mmwave(self):
        now = time.time()
        events = [
            SensorEvent(sensor_type="mmwave", area="kitchen", triggered_at=now),
            SensorEvent(sensor_type="motion", area="bedroom", triggered_at=now),
        ]
        scores = compute_room_confidence(events)
        assert scores["kitchen"] > scores["bedroom"]

    def test_old_event_scores_lower(self):
        now = time.time()
        events = [
            SensorEvent(sensor_type="mmwave", area="kitchen", triggered_at=now - 200),
        ]
        scores = compute_room_confidence(events, decay_timeout=300.0)
        assert 0.0 < scores["kitchen"] < 0.5

    def test_expired_event_excluded(self):
        now = time.time()
        events = [
            SensorEvent(sensor_type="motion", area="kitchen", triggered_at=now - 400),
        ]
        scores = compute_room_confidence(events, decay_timeout=300.0)
        assert "kitchen" not in scores or scores.get("kitchen", 0) == 0.0

    def test_empty_events_returns_empty(self):
        scores = compute_room_confidence([])
        assert scores == {}

    def test_no_area_events_ignored(self):
        now = time.time()
        events = [SensorEvent(sensor_type="motion", area="", triggered_at=now)]
        scores = compute_room_confidence(events)
        assert scores == {}

    def test_highest_score_kept_per_room(self):
        now = time.time()
        events = [
            SensorEvent(sensor_type="motion", area="kitchen", triggered_at=now),
            SensorEvent(sensor_type="mmwave", area="kitchen", triggered_at=now),
        ]
        scores = compute_room_confidence(events)
        # mmwave weight is 1.0, motion is 0.5; max kept
        assert scores["kitchen"] == pytest.approx(1.0, abs=0.01)

    def test_sensor_weights_defined(self):
        assert SENSOR_WEIGHTS["mmwave"] == 1.0
        assert SENSOR_WEIGHTS["occupancy"] == 0.8
        assert SENSOR_WEIGHTS["motion"] == 0.5

    def test_confidence_in_zero_one_range(self):
        now = time.time()
        events = [
            SensorEvent(sensor_type="mmwave", area="a", triggered_at=now),
            SensorEvent(sensor_type="occupancy", area="b", triggered_at=now - 100),
            SensorEvent(sensor_type="motion", area="c", triggered_at=now - 250),
        ]
        scores = compute_room_confidence(events, decay_timeout=300.0)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# PresenceDetector
# ---------------------------------------------------------------------------

class TestPresenceDetector:

    @patch("app.presence.detector.SettingsRepository")
    @patch("app.presence.detector.discover_sensors", new_callable=AsyncMock, return_value=[])
    async def test_initialize_loads_config(self, mock_discover, mock_settings):
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "presence.enabled": "true",
            "presence.decay_timeout": "600",
        }.get(k, d))
        ha = AsyncMock()
        detector = PresenceDetector(ha)
        await detector.initialize()
        assert detector._decay_timeout == 600.0
        assert detector._enabled is True

    @patch("app.presence.detector.SettingsRepository")
    @patch("app.presence.detector.discover_sensors", new_callable=AsyncMock, return_value=[])
    async def test_initialize_disabled(self, mock_discover, mock_settings):
        mock_settings.get_value = AsyncMock(side_effect=lambda k, d=None: {
            "presence.enabled": "false",
        }.get(k, d))
        ha = AsyncMock()
        detector = PresenceDetector(ha)
        await detector.initialize()
        assert detector._enabled is False

    def test_on_sensor_state_change_records_event(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        sensor = PresenceSensor(
            entity_id="binary_sensor.motion",
            friendly_name="Motion",
            sensor_type="motion",
            area="kitchen",
            state="on",
        )
        detector._sensors = [sensor]
        detector.on_sensor_state_change("binary_sensor.motion", "on", "kitchen")
        assert len(detector._events) == 1
        assert detector._events[0].area == "kitchen"

    def test_on_sensor_state_change_ignores_off(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        sensor = PresenceSensor(
            entity_id="binary_sensor.motion",
            friendly_name="Motion",
            sensor_type="motion",
            area="kitchen",
            state="on",
        )
        detector._sensors = [sensor]
        detector.on_sensor_state_change("binary_sensor.motion", "off", "kitchen")
        assert len(detector._events) == 0

    def test_on_sensor_state_change_ignores_unknown_sensor(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        detector._sensors = []
        detector.on_sensor_state_change("binary_sensor.unknown", "on", "room")
        assert len(detector._events) == 0

    def test_get_room_confidence_returns_dict(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        scores = detector.get_room_confidence()
        assert isinstance(scores, dict)

    def test_get_room_confidence_empty_when_disabled(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        detector._enabled = False
        scores = detector.get_room_confidence()
        assert scores == {}

    def test_get_most_likely_room_returns_highest(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        now = time.time()
        sensor_kitchen = PresenceSensor("s1", "s1", "mmwave", "kitchen", "on")
        sensor_bedroom = PresenceSensor("s2", "s2", "motion", "bedroom", "on")
        detector._sensors = [sensor_kitchen, sensor_bedroom]
        detector._events = [
            SensorEvent("mmwave", "kitchen", now),
            SensorEvent("motion", "bedroom", now),
        ]
        room = detector.get_most_likely_room()
        assert room == "kitchen"

    def test_get_most_likely_room_returns_none_when_empty(self):
        ha = MagicMock()
        detector = PresenceDetector(ha)
        assert detector.get_most_likely_room() is None
