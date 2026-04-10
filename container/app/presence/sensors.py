"""Sensor auto-discovery and mapping."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sensor types and their device classes
MOTION_CLASSES = {"motion"}
OCCUPANCY_CLASSES = {"occupancy", "presence"}
MMWAVE_KEYWORDS = {"mmwave", "mmw", "ld2410", "ld2450"}


@dataclass
class PresenceSensor:
    entity_id: str
    friendly_name: str
    sensor_type: str  # "motion", "occupancy", "mmwave"
    area: str | None
    state: str  # "on" / "off"


async def discover_sensors(ha_client) -> list[PresenceSensor]:
    """Query HA REST API for motion, occupancy, and mmWave sensors."""
    states = await ha_client.get_states()
    sensors = []
    for state in states:
        entity_id = state.get("entity_id", "")
        if not entity_id.startswith("binary_sensor."):
            continue
        attrs = state.get("attributes", {})
        device_class = attrs.get("device_class", "")
        friendly_name = attrs.get("friendly_name", "")
        area = attrs.get("area_id")

        sensor_type = None
        if device_class in MOTION_CLASSES:
            sensor_type = "motion"
        elif device_class in OCCUPANCY_CLASSES:
            sensor_type = "occupancy"
        # Check for mmWave by name heuristic
        if sensor_type is None:
            name_lower = friendly_name.lower()
            if any(kw in name_lower for kw in MMWAVE_KEYWORDS):
                sensor_type = "mmwave"

        if sensor_type:
            sensors.append(PresenceSensor(
                entity_id=entity_id,
                friendly_name=friendly_name,
                sensor_type=sensor_type,
                area=area,
                state=state.get("state", "off"),
            ))
    return sensors
