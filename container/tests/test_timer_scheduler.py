"""Tests for ``app.agents.timer_scheduler.TimerScheduler`` (0.26.0).

The scheduler owns every non-native timer. These tests verify
persistence, fire dispatch per kind, cancellation, and restart
recovery. They do NOT exercise any HA ``timer.*`` helper -- the helper
pool is gone.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.timer_scheduler import TimerScheduler
from app.db.repository import ScheduledTimersRepository

pytestmark = pytest.mark.asyncio


def _make_scheduler(*, gateway=None) -> tuple[TimerScheduler, MagicMock]:
    orchestrator_gateway = gateway or MagicMock()
    orchestrator_gateway.dispatch_background_event = AsyncMock()
    scheduler = TimerScheduler(
        ScheduledTimersRepository,
        orchestrator_gateway=orchestrator_gateway,
    )
    return scheduler, orchestrator_gateway


class TestSchedulePersistence:
    async def test_schedule_persists_row(self, db_repository):
        sched, _gateway = _make_scheduler()
        try:
            timer_id = await sched.schedule(
                logical_name="egg timer",
                kind="notification",
                duration_seconds=60,
                origin_device_id="dev-1",
                origin_area="kitchen",
                payload={"notification_message": "Eggs done!", "language": "de"},
            )
            assert timer_id
            rows = await ScheduledTimersRepository.list_pending()
            assert len(rows) == 1
            row = rows[0]
            assert row["logical_name"] == "egg timer"
            assert row["kind"] == "notification"
            assert row["origin_area"] == "kitchen"
            assert json.loads(row["payload_json"])["notification_message"] == "Eggs done!"
            assert json.loads(row["payload_json"])["language"] == "de"
            assert row["state"] == "pending"
        finally:
            await sched.stop()


class TestFireOnDeadline:
    async def test_schedule_fires_on_deadline(self, db_repository):
        sched, gateway = _make_scheduler()
        try:
            timer_id = await sched.schedule(
                logical_name="quick",
                kind="notification",
                duration_seconds=0,
                payload={"notification_message": "boom"},
            )
            for _ in range(20):
                await asyncio.sleep(0.02)
                row = await ScheduledTimersRepository.get(timer_id)
                if row and row["state"] == "fired":
                    break
            assert row["state"] == "fired"
            gateway.dispatch_background_event.assert_awaited_once()
            assert gateway.dispatch_background_event.await_args.args[0] == "timer_notification"
        finally:
            await sched.stop()


class TestCancellation:
    async def test_cancel_by_id(self, db_repository):
        sched, _gateway = _make_scheduler()
        try:
            timer_id = await sched.schedule(
                logical_name="long",
                kind="notification",
                duration_seconds=3600,
                payload={"notification_message": "x"},
            )
            count = await sched.cancel(id_=timer_id)
            assert count == 1
            row = await ScheduledTimersRepository.get(timer_id)
            assert row["state"] == "cancelled"
        finally:
            await sched.stop()

    async def test_cancel_by_logical_name(self, db_repository):
        sched, _gateway = _make_scheduler()
        try:
            id1 = await sched.schedule(
                logical_name="dupe",
                kind="notification",
                duration_seconds=3600,
                payload={"notification_message": "a"},
            )
            id2 = await sched.schedule(
                logical_name="dupe",
                kind="notification",
                duration_seconds=3600,
                payload={"notification_message": "b"},
            )
            count = await sched.cancel(logical_name="dupe")
            assert count == 2
            for tid in (id1, id2):
                row = await ScheduledTimersRepository.get(tid)
                assert row["state"] == "cancelled"
        finally:
            await sched.stop()


class TestList:
    async def test_list_filters_by_area(self, db_repository):
        sched, _gateway = _make_scheduler()
        try:
            await sched.schedule(
                logical_name="a",
                kind="notification",
                duration_seconds=3600,
                origin_area="kitchen",
                payload={"notification_message": "x"},
            )
            await sched.schedule(
                logical_name="b",
                kind="notification",
                duration_seconds=3600,
                origin_area="bedroom",
                payload={"notification_message": "x"},
            )
            await sched.schedule(
                logical_name="c",
                kind="notification",
                duration_seconds=3600,
                origin_area="kitchen",
                payload={"notification_message": "x"},
            )
            kitchen = await sched.list(area="kitchen")
            assert {r["logical_name"] for r in kitchen} == {"a", "c"}
        finally:
            await sched.stop()

    async def test_list_filters_by_kind(self, db_repository):
        sched, _gateway = _make_scheduler()
        try:
            await sched.schedule(
                logical_name="kitchen alarm",
                kind="alarm",
                duration_seconds=3600,
                origin_area="kitchen",
                payload={"alarm_label": "Kitchen Alarm"},
            )
            await sched.schedule(
                logical_name="kitchen timer",
                kind="notification",
                duration_seconds=3600,
                origin_area="kitchen",
                payload={"notification_message": "done"},
            )

            alarms = await sched.list(area="kitchen", kinds={"alarm"})
            assert len(alarms) == 1
            assert alarms[0]["kind"] == "alarm"
            assert alarms[0]["logical_name"] == "kitchen alarm"
        finally:
            await sched.stop()


class TestRestartRecovery:
    async def test_restart_recovery(self, db_repository):
        # Scheduler #1: create a long-pending timer and stop without firing.
        s1, _gateway = _make_scheduler()
        timer_id = await s1.schedule(
            logical_name="rehydrate-me",
            kind="notification",
            duration_seconds=3600,
            payload={"notification_message": "x"},
        )
        await s1.stop()

        # Scheduler #2: must rehydrate the pending row.
        s2, _gateway2 = _make_scheduler()
        try:
            await s2.start()
            assert timer_id in s2._tasks
        finally:
            await s2.stop()

    async def test_overdue_timer_fires_on_recovery(self, db_repository):
        # Insert an already-overdue row directly (mimics a process that
        # crashed before its asyncio task could fire).
        await ScheduledTimersRepository.insert(
            id="overdue-id",
            logical_name="overdue",
            kind="notification",
            created_at=int(time.time()) - 100,
            fires_at=int(time.time()) - 10,
            duration_seconds=10,
            origin_device_id=None,
            origin_area=None,
            payload_json=json.dumps({"notification_message": "late"}),
        )
        sched, gateway = _make_scheduler()
        try:
            await sched.start()
            row = await ScheduledTimersRepository.get("overdue-id")
            assert row["state"] == "fired"
            gateway.dispatch_background_event.assert_awaited_once()
            assert gateway.dispatch_background_event.await_args.args[0] == "timer_notification"
        finally:
            await sched.stop()


class TestKindDispatch:
    async def test_plain_and_notification_dispatch_background_event(self, db_repository):
        sched, gateway = _make_scheduler()
        try:
            tid = await sched.schedule(
                logical_name="x",
                kind="plain",
                duration_seconds=0,
                origin_device_id="device-123",
                origin_area="kitchen",
                payload={"language": "de"},
            )
            for _ in range(20):
                await asyncio.sleep(0.02)
                row = await ScheduledTimersRepository.get(tid)
                if row and row["state"] == "fired":
                    break
            gateway.dispatch_background_event.assert_awaited_once()
            assert gateway.dispatch_background_event.await_args.args[0] == "timer_notification"
            payload = gateway.dispatch_background_event.await_args.args[1]
            assert payload["origin_device_id"] == "device-123"
            assert payload["origin_area"] == "kitchen"
            assert payload["language"] == "de"
        finally:
            await sched.stop()

    async def test_delayed_action_fires_target_service(self, db_repository):
        sched, gateway = _make_scheduler()
        try:
            tid = await sched.schedule(
                logical_name="lights off",
                kind="delayed_action",
                duration_seconds=0,
                payload={
                    "target_entity": "light.kitchen",
                    "target_action": "light/turn_off",
                },
            )
            for _ in range(20):
                await asyncio.sleep(0.02)
                row = await ScheduledTimersRepository.get(tid)
                if row and row["state"] == "fired":
                    break
            gateway.dispatch_background_event.assert_awaited_once()
            assert gateway.dispatch_background_event.await_args.args[0] == "delayed_action"
            assert gateway.dispatch_background_event.await_args.args[1]["target_entity"] == "light.kitchen"
        finally:
            await sched.stop()

    async def test_sleep_kind_calls_media_stop(self, db_repository):
        sched, gateway = _make_scheduler()
        try:
            tid = await sched.schedule(
                logical_name="bedtime",
                kind="sleep",
                duration_seconds=0,
                payload={"media_player": "media_player.bedroom"},
            )
            for _ in range(20):
                await asyncio.sleep(0.02)
                row = await ScheduledTimersRepository.get(tid)
                if row and row["state"] == "fired":
                    break
            gateway.dispatch_background_event.assert_awaited_once()
            assert gateway.dispatch_background_event.await_args.args[0] == "sleep_media_stop"
            assert gateway.dispatch_background_event.await_args.args[1]["media_player"] == "media_player.bedroom"
        finally:
            await sched.stop()

    async def test_alarm_kind_dispatches_alarm_notification_with_origin_metadata(self, db_repository):
        sched, gateway = _make_scheduler()
        try:
            tid = await sched.schedule(
                logical_name="Morning Alarm",
                kind="alarm",
                duration_seconds=0,
                origin_device_id="device-123",
                origin_area="bedroom",
                payload={
                    "alarm_label": "Morning Alarm",
                    "language": "de",
                    "media_player": "media_player.bedroom",
                },
            )
            for _ in range(20):
                await asyncio.sleep(0.02)
                row = await ScheduledTimersRepository.get(tid)
                if row and row["state"] == "fired":
                    break

            gateway.dispatch_background_event.assert_awaited_once()
            assert gateway.dispatch_background_event.await_args.args[0] == "alarm_notification"
            payload = gateway.dispatch_background_event.await_args.args[1]
            assert payload["alarm_name"] == "Morning Alarm"
            assert payload["entity_id"].startswith("agenthub_alarm:")
            assert payload["origin_device_id"] == "device-123"
            assert payload["origin_area"] == "bedroom"
            assert payload["language"] == "de"
            assert payload["media_player"] == "media_player.bedroom"
        finally:
            await sched.stop()
