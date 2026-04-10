"""Room-level presence confidence scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass

# Weight by sensor type
SENSOR_WEIGHTS = {
    "mmwave": 1.0,
    "occupancy": 0.8,
    "motion": 0.5,
}


@dataclass
class SensorEvent:
    sensor_type: str
    area: str
    triggered_at: float  # time.time() epoch


def compute_room_confidence(
    events: list[SensorEvent],
    decay_timeout: float = 300.0,
) -> dict[str, float]:
    """Compute per-room presence confidence from sensor events.

    Args:
        events: List of recent sensor events.
        decay_timeout: Seconds after which confidence decays to 0.

    Returns:
        Dict of {area: confidence} where confidence is 0.0-1.0.
    """
    now = time.time()
    room_scores: dict[str, float] = {}

    for event in events:
        if not event.area:
            continue
        age = now - event.triggered_at
        if age > decay_timeout:
            continue
        # Recency factor: 1.0 at trigger time, 0.0 at decay_timeout
        recency = max(0.0, 1.0 - (age / decay_timeout))
        weight = SENSOR_WEIGHTS.get(event.sensor_type, 0.5)
        score = weight * recency
        # Keep the highest score per room
        room_scores[event.area] = max(room_scores.get(event.area, 0.0), score)

    return room_scores
