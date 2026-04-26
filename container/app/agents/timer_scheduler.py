"""AgentHub-managed timer scheduler.

Owns every non-native timer (notification, delayed_action, sleep,
snooze, internal plain). Persists state in SQLite via
``ScheduledTimersRepository`` so timers survive container restart, and
runs one ``asyncio.Task`` per pending timer for wall-clock firing
independent of any HA timer.* helper.

Replaces the obsolete HA ``timer.*`` helper-pool model that lived in
``timer_executor.py`` and ``delayed_tasks.py`` prior to 0.26.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from app.a2a.orchestrator_gateway import OrchestratorGateway
from app.db.repository import ScheduledTimersRepository

logger = logging.getLogger(__name__)


_VALID_KINDS = frozenset({"plain", "notification", "delayed_action", "sleep", "snooze", "alarm"})
_STARTUP_RECOVERY_RETRY_DELAY_SECONDS = 2.0


class TimerScheduler:
    """In-process scheduler with persisted state.

    Each pending timer is backed by a row in ``scheduled_timers`` and an
    ``asyncio.Task`` that sleeps until ``fires_at`` then dispatches the
    kind-specific fire callback.
    """

    def __init__(
        self,
        repo: type[ScheduledTimersRepository] | ScheduledTimersRepository = ScheduledTimersRepository,
        *,
        orchestrator_gateway: OrchestratorGateway | None = None,
    ) -> None:
        self._repo = repo
        self._orchestrator_gateway = orchestrator_gateway
        self._tasks: dict[str, asyncio.Task] = {}
        self._by_logical: dict[str, list[str]] = {}
        self._startup_recovery_task: asyncio.Task | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Rehydrate pending timers from the DB.

        Overdue timers fire immediately during startup. All other
        pending timers get an asyncio task that sleeps until their
        ``fires_at``.
        """
        if self._started:
            return
        self._started = True
        rehydrated = 0
        fired_on_recovery = 0
        try:
            rows = await self._repo.list_pending()
        except Exception:
            logger.error("TimerScheduler.start: failed to load pending timers", exc_info=True)
            self._schedule_startup_recovery_retry()
            rows = []
        rehydrated, fired_on_recovery = await self._rehydrate_rows(rows)
        logger.info(
            "TimerScheduler started: rehydrated=%d fired_on_recovery=%d",
            rehydrated,
            fired_on_recovery,
        )

    async def stop(self) -> None:
        """Cancel all in-flight timer tasks. DB rows remain pending."""
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        startup_recovery = self._startup_recovery_task
        self._startup_recovery_task = None
        if startup_recovery and not startup_recovery.done():
            startup_recovery.cancel()
            await asyncio.gather(startup_recovery, return_exceptions=True)
        self._tasks.clear()
        self._by_logical.clear()
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def schedule(
        self,
        *,
        logical_name: str,
        kind: str,
        duration_seconds: int,
        origin_device_id: str | None = None,
        origin_area: str | None = None,
        payload: dict | None = None,
    ) -> str:
        """Persist a new timer row and start its firing task."""
        if kind not in _VALID_KINDS:
            raise ValueError(f"Unknown timer kind: {kind}")
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        timer_id = uuid.uuid4().hex
        now = int(time.time())
        fires_at = now + int(duration_seconds)
        payload_json = json.dumps(payload or {})
        await self._repo.insert(
            id=timer_id,
            logical_name=logical_name,
            kind=kind,
            created_at=now,
            fires_at=fires_at,
            duration_seconds=int(duration_seconds),
            origin_device_id=origin_device_id,
            origin_area=origin_area,
            payload_json=payload_json,
        )
        row = {
            "id": timer_id,
            "logical_name": logical_name,
            "kind": kind,
            "created_at": now,
            "fires_at": fires_at,
            "duration_seconds": int(duration_seconds),
            "origin_device_id": origin_device_id,
            "origin_area": origin_area,
            "payload_json": payload_json,
            "state": "pending",
        }
        self._spawn_task(row)
        return timer_id

    async def cancel(
        self,
        *,
        id_: str | None = None,
        logical_name: str | None = None,
        area: str | None = None,
    ) -> int:
        """Cancel by id or by logical_name (optionally scoped to area).

        Returns the number of timers cancelled.
        """
        now = int(time.time())
        if id_:
            row = await self._repo.get(id_)
            if not row or row.get("state") != "pending":
                return 0
            await self._repo.mark_cancelled(id_, now)
            self._cancel_task(id_)
            return 1
        if not logical_name:
            return 0
        rows = await self._repo.list_pending_for(logical_name=logical_name, area=area)
        if not rows:
            return 0
        count = 0
        for row in rows:
            await self._repo.mark_cancelled(row["id"], now)
            self._cancel_task(row["id"])
            count += 1
        return count

    async def list(
        self,
        *,
        logical_name: str | None = None,
        area: str | None = None,
        kinds: set[str] | frozenset[str] | None = None,
    ) -> list[dict]:
        """Return pending timers, optionally filtered by logical_name and/or area."""
        return await self._repo.list_pending_for(logical_name=logical_name, area=area, kinds=kinds)

    async def reschedule(
        self,
        id_: str,
        *,
        logical_name: str | None = None,
        new_fires_at: int | None = None,
        new_duration_seconds: int | None = None,
    ) -> bool:
        """Update a pending timer/alarm in-place and restart its asyncio task.

        Strategy: update the DB row via the repository, then cancel the
        existing asyncio task and spawn a replacement task from the updated
        row. The row ID is preserved (no identity churn).

        Returns ``True`` if the timer was found, was still pending, and was
        updated. Returns ``False`` if no matching pending row exists.
        """
        row = await self._repo.get(id_)
        if not row or row.get("state") != "pending":
            return False

        updated = await self._repo.update_scheduled_timer(
            id_,
            logical_name=logical_name,
            fires_at=new_fires_at,
            duration_seconds=new_duration_seconds,
        )
        if not updated:
            return False

        updated_row = dict(row)
        if logical_name is not None:
            old_key = (row.get("logical_name") or "").lower()
            new_key = logical_name.lower()
            if old_key != new_key:
                old_list = self._by_logical.get(old_key, [])
                if id_ in old_list:
                    old_list.remove(id_)
                self._by_logical.setdefault(new_key, []).append(id_)
            updated_row["logical_name"] = logical_name
        if new_fires_at is not None:
            updated_row["fires_at"] = int(new_fires_at)
        if new_duration_seconds is not None:
            updated_row["duration_seconds"] = int(new_duration_seconds)

        # Keep cancellation semantics centralized in _cancel_task.
        self._cancel_task(id_)
        self._spawn_task(updated_row)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_task(self, row: dict) -> None:
        timer_id = row["id"]
        if timer_id in self._tasks:
            return
        task = asyncio.create_task(self._run(row), name=f"timer-{timer_id}")
        self._tasks[timer_id] = task
        self._by_logical.setdefault((row["logical_name"] or "").lower(), []).append(timer_id)

    def _schedule_startup_recovery_retry(self) -> None:
        existing = self._startup_recovery_task
        if existing is not None and not existing.done():
            return
        self._startup_recovery_task = asyncio.create_task(
            self._startup_recovery_retry(),
            name="timer-startup-recovery-retry",
        )

    async def _startup_recovery_retry(self) -> None:
        try:
            await asyncio.sleep(_STARTUP_RECOVERY_RETRY_DELAY_SECONDS)
            rows = await self._repo.list_pending()
            rehydrated, fired_on_recovery = await self._rehydrate_rows(rows)
            logger.info(
                "TimerScheduler startup recovery retry: rehydrated=%d fired_on_recovery=%d",
                rehydrated,
                fired_on_recovery,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("TimerScheduler startup recovery retry failed", exc_info=True)

    async def _rehydrate_rows(self, rows: list[dict]) -> tuple[int, int]:
        rehydrated = 0
        fired_on_recovery = 0
        now = int(time.time())
        for row in rows:
            timer_id = row.get("id")
            if not timer_id or timer_id in self._tasks:
                continue
            if int(row["fires_at"]) <= now:
                try:
                    await self._fire(row)
                    await self._repo.mark_fired(timer_id, now)
                    fired_on_recovery += 1
                except Exception:
                    logger.error(
                        "TimerScheduler.start: fire-on-recovery failed for %s",
                        row.get("id"),
                        exc_info=True,
                    )
            else:
                self._spawn_task(row)
                rehydrated += 1
        return rehydrated, fired_on_recovery

    def _cancel_task(self, timer_id: str) -> None:
        task = self._tasks.pop(timer_id, None)
        if task and not task.done():
            task.cancel()
        for ids in self._by_logical.values():
            if timer_id in ids:
                ids.remove(timer_id)

    async def _run(self, row: dict) -> None:
        timer_id = row["id"]
        try:
            delay = max(0.0, float(row["fires_at"]) - time.time())
            if delay > 0:
                await asyncio.sleep(delay)
            await self._fire(row)
            await self._repo.mark_fired(timer_id, int(time.time()))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Timer %s fire failed", timer_id, exc_info=True)
            try:
                await self._repo.mark_fired(timer_id, int(time.time()))
            except Exception:
                logger.error("Timer %s mark_fired failed", timer_id, exc_info=True)
        finally:
            self._tasks.pop(timer_id, None)
            for ids in self._by_logical.values():
                if timer_id in ids:
                    ids.remove(timer_id)

    async def _fire(self, row: dict) -> None:
        """Dispatch the kind-specific fire action.

        Errors are logged; the timer is still marked fired by the
        caller. Exact-once semantics surfaced to logs.
        """
        kind = row["kind"]
        try:
            payload = json.loads(row.get("payload_json") or "{}")
        except json.JSONDecodeError:
            payload = {}

        logical_name = row.get("logical_name") or ""
        origin_device_id = row.get("origin_device_id")
        origin_area = row.get("origin_area")
        duration_seconds = int(row.get("duration_seconds") or 0)
        duration_str = _seconds_to_hms(duration_seconds)
        language = payload.get("language")
        gateway = self._orchestrator_gateway
        if gateway is None and kind != "snooze":
            logger.warning("TimerScheduler fire skipped for %s: no orchestrator gateway", row["id"])
            return

        if kind in ("plain", "notification"):
            message = payload.get("notification_message") if kind == "notification" else None
            synthetic_entity_id = f"agenthub_internal:{row['id']}"
            display_name = logical_name
            if message:
                display_name = f"{logical_name}: {message}" if logical_name else message
            await gateway.dispatch_background_event(
                "timer_notification",
                {
                    "timer_name": display_name,
                    "entity_id": synthetic_entity_id,
                    "media_player": payload.get("media_player"),
                    "origin_device_id": origin_device_id,
                    "origin_area": origin_area,
                    "duration": duration_str,
                    "language": language,
                },
                description=f"Dispatch timer notification for {display_name or 'timer'}",
            )
            return

        if kind == "alarm":
            alarm_name = (payload.get("alarm_label") or logical_name or "alarm").strip() or "alarm"
            synthetic_entity_id = f"agenthub_alarm:{row['id']}"
            await gateway.dispatch_background_event(
                "alarm_notification",
                {
                    "alarm_name": alarm_name,
                    "entity_id": synthetic_entity_id,
                    "media_player": payload.get("media_player"),
                    "origin_device_id": origin_device_id,
                    "origin_area": origin_area,
                    "language": language,
                },
                description=f"Dispatch alarm notification for {alarm_name}",
            )
            return

        if kind == "delayed_action":
            target_entity = payload.get("target_entity") or ""
            target_action = payload.get("target_action") or ""
            if not target_entity or "/" not in target_action:
                logger.error("delayed_action timer %s missing target_entity/target_action", row["id"])
                return
            await gateway.dispatch_background_event(
                "delayed_action",
                {
                    "target_entity": target_entity,
                    "target_action": target_action,
                },
                description=f"Execute delayed action {target_action} for {target_entity}",
            )
            return

        if kind == "sleep":
            media_player = payload.get("media_player") or ""
            if not media_player:
                logger.error("sleep timer %s missing media_player", row["id"])
                return
            await gateway.dispatch_background_event(
                "sleep_media_stop",
                {"media_player": media_player},
                description=f"Stop media playback for {media_player}",
            )
            return

        if kind == "snooze":
            snooze_seconds = int(payload.get("snooze_seconds") or duration_seconds)
            await self.schedule(
                logical_name=logical_name,
                kind="plain",
                duration_seconds=snooze_seconds,
                origin_device_id=origin_device_id,
                origin_area=origin_area,
                payload={"snoozed_from": row["id"], "language": language},
            )
            return

        logger.warning("Unknown timer kind for %s: %s", row["id"], kind)


def _seconds_to_hms(total: int) -> str:
    if total <= 0:
        return "00:00:00"
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_timer_scheduler(app: Any = None) -> TimerScheduler | None:
    """Return the scheduler stored on ``app.state.timer_scheduler``, or None."""
    if app is None:
        return None
    return getattr(app.state, "timer_scheduler", None)
