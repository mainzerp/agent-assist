"""Background task tracking utilities.

Holds strong references to background tasks so they are not garbage
collected mid-flight (Python's asyncio holds only weak refs to tasks).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

_pending: set[asyncio.Task] = set()
_log = logging.getLogger(__name__)


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task:
    """Schedule ``coro`` as a tracked background task.

    The task is stored in a module-level set until completion so it
    cannot be silently dropped by the GC. Exceptions raised inside the
    coroutine are logged with traceback and do not propagate.
    """
    task = asyncio.create_task(coro, name=name)
    _pending.add(task)

    def _done(t: asyncio.Task) -> None:
        _pending.discard(t)
        if not t.cancelled() and t.exception() is not None:
            _log.error("Background task %s failed", t.get_name(), exc_info=t.exception())

    task.add_done_callback(_done)
    return task
