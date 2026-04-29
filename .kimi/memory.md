# Orchestrator Memory

> This file is the orchestrator's long-term memory across sessions.
> Read it at the start of every session to recall context.
> Append new learnings, patterns, and reminders as they emerge.
> Never delete existing entries unless they are explicitly superseded.

## Lessons Learned

- 2026-04-29: The container's pre-existing test failures (timer reroute, orchestrator routing cache, health endpoint, entity index rebuild, registry invalidation) have all been fixed. Future work on the container backend should run `pytest tests/ -n auto` before pushing to avoid regressions.
- 2026-04-29: `docker-compose_local.yml` is the correct compose file for local development (service `agent-assist`, volume `agent-assist-data`). The root `docker-compose.yml` pulls from GHCR and uses different names.
- 2026-04-29: When importing `custom_components.ha_agenthub` in container tests, mock `voluptuous` and `homeassistant.helpers.selector` in `conftest.py` before any import, or the import chain will fail.

## Recurring Patterns / Gotchas

### Container Tests
- Any test that mocks `ha_client` and calls `runtime_setup._prime_entity_index` or `_refresh_registry_entities` must provide `ha_client.get_hidden_entity_ids = AsyncMock(return_value=set())`.
- `asyncio.create_task` mocks in tests must accept `**kwargs` because Python 3.12 passes `name=...` by default.

### Integration (HA) Tests
- The integration's `config_flow.py` imports `voluptuous` and `homeassistant.helpers.selector` at module level. Any test that imports `conversation.py` (which imports `__init__.py`, which imports `config_flow.py`) will fail unless these modules are pre-mocked in `sys.modules`.

### Lint / CI
- `ruff check` and `ruff format` must both pass before pushing. The CI runs both.
- Container tests use `pytest-xdist` (`-n auto`) in CI but not necessarily locally.

## Active Conventions (evolving)

- Use `AsyncMock` for all async mocks, not plain `MagicMock`.
- When fixing a test, check if the fix belongs in the test (mock missing) or in production code (actual bug).
- The orchestrator's `_classify` method must return `user_text` on routing cache hits, not the stale `condensed_task`.

## Open / Carried Over

- None currently.

## Critical Do-Nots

- **NEVER delete Docker volumes** (`docker compose down -v`, `docker volume rm`, etc.) unless the user *explicitly* requests it. This destroys the SQLite database, ChromaDB data, API keys, HA connection config, and all cached state. If a container fails to start due to data corruption, prefer `docker compose down` (without `-v`) and container recreation, or manual cleanup of specific files inside the volume. Losing a volume forces a full re-setup of the container.
