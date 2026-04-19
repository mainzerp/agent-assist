"""In-memory async delayed task manager for post-timer actions.

Manages background asyncio tasks that execute a callback after a timer
completes. Tasks are tracked by timer entity_id and can be cancelled.

Limitations:
- Tasks are lost on process restart (acceptable for v1).
- Uses polling to detect timer completion (robust to pause/resume).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# How often to poll HA for timer state (seconds)
_POLL_INTERVAL = 3.0


@dataclass
class DelayedTask:
    """A pending action to execute after a timer finishes."""

    timer_entity_id: str
    description: str
    callback: Callable[[], Awaitable[None]]
    task: asyncio.Task | None = field(default=None, repr=False)


class DelayedTaskManager:
    """Manages async tasks that fire when a timer entity transitions to idle."""

    def __init__(self) -> None:
        self._tasks: dict[str, DelayedTask] = {}  # keyed by timer_entity_id
        self._ha_client: Any = None

    def set_ha_client(self, ha_client: Any) -> None:
        """Set the HA client reference (called at startup)."""
        self._ha_client = ha_client

    def schedule(
        self,
        timer_entity_id: str,
        description: str,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Schedule a callback to run when the timer entity goes idle.

        If a task already exists for this entity, it is cancelled and replaced.
        """
        self.cancel(timer_entity_id)
        dt = DelayedTask(
            timer_entity_id=timer_entity_id,
            description=description,
            callback=callback,
        )
        dt.task = asyncio.create_task(self._poll_and_execute(dt))
        self._tasks[timer_entity_id] = dt
        logger.info("Scheduled delayed task for %s: %s", timer_entity_id, description)

    def cancel(self, timer_entity_id: str) -> bool:
        """Cancel a pending delayed task. Returns True if one was cancelled."""
        dt = self._tasks.pop(timer_entity_id, None)
        if dt and dt.task and not dt.task.done():
            dt.task.cancel()
            logger.info("Cancelled delayed task for %s", timer_entity_id)
            return True
        return False

    def get_pending(self) -> list[dict]:
        """Return list of pending delayed tasks as dicts."""
        return [
            {"timer_entity_id": dt.timer_entity_id, "description": dt.description}
            for dt in self._tasks.values()
            if dt.task and not dt.task.done()
        ]

    async def _poll_and_execute(self, dt: DelayedTask) -> None:
        """Poll timer state until idle, then execute callback."""
        if not self._ha_client:
            logger.warning("No HA client for delayed task on %s", dt.timer_entity_id)
            return

        try:
            # Initial wait -- timer was just started, avoid immediate false idle
            await asyncio.sleep(_POLL_INTERVAL)

            while True:
                try:
                    state_resp = await self._ha_client.get_state(dt.timer_entity_id)
                    if not state_resp:
                        logger.warning("Timer entity %s not found, aborting delayed task", dt.timer_entity_id)
                        return
                    state = state_resp.get("state", "")
                    if state == "idle":
                        break
                except Exception:
                    logger.warning("Poll failed for %s, retrying", dt.timer_entity_id, exc_info=True)
                await asyncio.sleep(_POLL_INTERVAL)

            # Timer finished -- execute callback
            logger.info("Timer %s finished, executing delayed task: %s", dt.timer_entity_id, dt.description)
            await dt.callback()
        except asyncio.CancelledError:
            logger.debug("Delayed task for %s was cancelled", dt.timer_entity_id)
        except Exception:
            logger.error("Delayed task failed for %s", dt.timer_entity_id, exc_info=True)
        finally:
            self._tasks.pop(dt.timer_entity_id, None)


# Module-level singleton
delayed_task_manager = DelayedTaskManager()
