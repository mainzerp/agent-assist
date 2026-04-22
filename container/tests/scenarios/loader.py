"""YAML scenario loader and HA snapshot loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .types import (
    EntityOverride,
    Expected,
    ExpectedCall,
    ExpectedError,
    FollowUpTurn,
    LlmReplies,
    Preconditions,
    Scenario,
    ScenarioContext,
)

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_SNAPSHOT_ROOT = _DATA_ROOT / "ha_snapshots"
_SCENARIO_ROOT = _DATA_ROOT / "scenarios"


def snapshot_root() -> Path:
    return _SNAPSHOT_ROOT


def scenario_root() -> Path:
    return _SCENARIO_ROOT


def list_scenario_files() -> list[Path]:
    if not _SCENARIO_ROOT.exists():
        return []
    return sorted(_SCENARIO_ROOT.glob("**/*.yaml"))


def load_snapshot(name: str) -> dict[str, Any]:
    """Return ``{"states": [...], "areas": {...}, "devices": {...}, "config": {...}}``."""
    base = _SNAPSHOT_ROOT
    states_path = base / f"{name}.json"
    areas_path = base / f"{name}.areas.json"
    devices_path = base / f"{name}.devices.json"
    config_path = base / f"{name}.config.json"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    areas = json.loads(areas_path.read_text(encoding="utf-8")) if areas_path.exists() else {}
    devices = json.loads(devices_path.read_text(encoding="utf-8")) if devices_path.exists() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {"states": states, "areas": areas, "devices": devices, "config": config}


def _expected_from_dict(data: dict[str, Any] | None) -> Expected:
    data = data or {}
    calls = []
    for c in data.get("service_calls", []) or []:
        calls.append(
            ExpectedCall(
                domain=c["domain"],
                service=c["service"],
                target_entity=c.get("target_entity"),
                service_data_keys=list(c.get("service_data_keys", []) or []),
                service_data=dict(c.get("service_data", {}) or {}),
            )
        )
    err = None
    if data.get("error"):
        err = ExpectedError(code=data["error"]["code"])
    return Expected(
        routed_agent=data.get("routed_agent"),
        service_calls=calls,
        speech_contains=list(data.get("speech_contains", []) or []),
        speech_excludes=list(data.get("speech_excludes", []) or []),
        action_executed=data.get("action_executed"),
        error=err,
        allow_extra_calls=bool(data.get("allow_extra_calls", False)),
    )


def _llm_from_dict(data: dict[str, Any] | None) -> LlmReplies:
    data = data or {}
    agents_raw = data.get("agents") or {}
    agents: dict[str, list[str]] = {}
    for k, v in agents_raw.items():
        if isinstance(v, list):
            agents[k] = [str(x) for x in v]
        else:
            agents[k] = [str(v)]
    classify = data.get("classify")
    return LlmReplies(classify=str(classify) if classify is not None else None, agents=agents)


def _ctx_from_dict(data: dict[str, Any] | None) -> ScenarioContext:
    data = data or {}
    return ScenarioContext(
        source=data.get("source", "ha"),
        area_id=data.get("area_id"),
        area_name=data.get("area_name"),
        device_id=data.get("device_id"),
        device_name=data.get("device_name"),
        conversation_id=data.get("conversation_id"),
        user_id=data.get("user_id"),
    )


def _preconditions_from_dict(data: dict[str, Any] | None) -> Preconditions:
    data = data or {}
    overrides_raw = data.get("entity_overrides", []) or []
    overrides = [
        EntityOverride(
            entity_id=ov["entity_id"],
            state=ov.get("state"),
            attributes=dict(ov.get("attributes", {}) or {}),
        )
        for ov in overrides_raw
    ]
    settings_raw = data.get("settings", {}) or {}
    settings = {str(k): str(v) for k, v in settings_raw.items()}
    return Preconditions(
        entity_overrides=overrides,
        settings=settings,
        send_device_mappings=list(data.get("send_device_mappings", []) or []),
        frozen_time=data.get("frozen_time"),
    )


def load_scenario(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    request = raw.get("request") or {}
    text = request.get("text", "")
    follow_up_raw = raw.get("follow_up") or []
    follow_up: list[FollowUpTurn] = []
    for turn in follow_up_raw:
        treq = turn.get("request") or {}
        follow_up.append(
            FollowUpTurn(
                text=treq.get("text", ""),
                llm=_llm_from_dict(turn.get("llm")),
                expected=_expected_from_dict(turn.get("expected")),
            )
        )
    return Scenario(
        id=raw.get("id", path.stem),
        agent=raw.get("agent", ""),
        description=raw.get("description", ""),
        snapshot=raw.get("snapshot", "home_default"),
        language=raw.get("language", "en"),
        request_text=text,
        context=_ctx_from_dict(raw.get("context")),
        preconditions=_preconditions_from_dict(raw.get("preconditions")),
        llm=_llm_from_dict(raw.get("llm")),
        expected=_expected_from_dict(raw.get("expected")),
        follow_up=follow_up,
        path=str(path),
        xfail=raw.get("xfail"),
    )


def load_all_scenarios() -> list[Scenario]:
    return [load_scenario(p) for p in list_scenario_files()]
