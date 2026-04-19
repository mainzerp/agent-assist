"""Tests for app.util.tasks.spawn background task tracker."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from app.util import tasks as tasks_mod


@pytest.mark.asyncio
async def test_spawn_tracks_pending_until_done():
    started = asyncio.Event()
    release = asyncio.Event()

    async def _work() -> None:
        started.set()
        await release.wait()

    tasks_mod._pending.clear()
    task = tasks_mod.spawn(_work(), name="unit-pending")
    await started.wait()
    assert task in tasks_mod._pending
    release.set()
    await task
    # Allow done callback to run
    await asyncio.sleep(0)
    assert task not in tasks_mod._pending


@pytest.mark.asyncio
async def test_spawn_logs_exception_and_does_not_crash(caplog):
    async def _bad() -> None:
        raise RuntimeError("boom")

    tasks_mod._pending.clear()
    with caplog.at_level(logging.ERROR, logger="app.util.tasks"):
        task = tasks_mod.spawn(_bad(), name="unit-bad")
        with contextlib.suppress(RuntimeError):
            await task
        await asyncio.sleep(0)

    assert any("Background task unit-bad failed" in r.message for r in caplog.records)
    assert task not in tasks_mod._pending
