"""Presence detection engine."""

from __future__ import annotations

import logging
import time

from app.db.repository import SettingsRepository
from app.presence.sensors import PresenceSensor, discover_sensors
from app.presence.scoring import SensorEvent, compute_room_confidence

logger = logging.getLogger(__name__)


class PresenceDetector:
    """Aggregates sensor data for room-level presence detection."""

    def __init__(self, ha_client) -> None:
        self._ha_client = ha_client
        self._events: list[SensorEvent] = []
        self._sensors: list[PresenceSensor] = []
        self._decay_timeout: float = 300.0
        self._enabled: bool = True

    async def initialize(self) -> None:
        """Load config and discover sensors."""
        self._enabled = (await SettingsRepository.get_value("presence.enabled", "true")) == "true"
        if not self._enabled:
            logger.info("Presence detection disabled")
            return
        timeout_str = await SettingsRepository.get_value("presence.decay_timeout", "300")
        self._decay_timeout = float(timeout_str)
        try:
            self._sensors = await discover_sensors(self._ha_client)
            logger.info("Discovered %d presence sensors", len(self._sensors))
        except Exception:
            logger.warning("Failed to discover presence sensors", exc_info=True)

    def on_sensor_state_change(self, entity_id: str, new_state: str, area: str | None) -> None:
        """Called when a presence sensor state changes. Records event if 'on'."""
        if new_state != "on":
            return
        sensor = next((s for s in self._sensors if s.entity_id == entity_id), None)
        if not sensor:
            return
        self._events.append(SensorEvent(
            sensor_type=sensor.sensor_type,
            area=area or sensor.area or "",
            triggered_at=time.time(),
        ))
        # Prune old events
        cutoff = time.time() - self._decay_timeout
        self._events = [e for e in self._events if e.triggered_at > cutoff]

    def get_room_confidence(self) -> dict[str, float]:
        """Return current per-room presence confidence map."""
        if not self._enabled:
            return {}
        return compute_room_confidence(self._events, self._decay_timeout)

    def get_most_likely_room(self) -> str | None:
        """Return the room with highest presence confidence, or None."""
        scores = self.get_room_confidence()
        if not scores:
            return None
        return max(scores, key=scores.get)
