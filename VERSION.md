# Version

**Current Version:** 0.21.2

## Version History

### 0.21.2 (PATCH) -- Orchestrator condensed-task hardening

Fixes a routing regression where the condensed task forwarded to a
specialized agent could contain duplicated classification fragments
(e.g. "climate-agent (96%): living room temperatureclimate-agent
(96%): ...") and where non-English entity names such as "Wohnzimmer"
were translated to English. The orchestrator parser now strips
embedded `<known-agent> (NN%):` fragments and collapses verbatim
repetitions; the classification prompt receives a per-request language
hint that instructs the LLM to copy localized entity names verbatim
(few-shot examples remain English-only to keep the prompt universal);
the routing cache defensively rejects pre-existing entries whose
condensed_task still contains an embedded fragment so legacy
corruption self-heals.

Compatibility: no API or schema changes; existing routing-cache
entries that pass the new validation continue to be served.

### 0.21.1 (PATCH) -- Lint cleanup

Pure lint cleanup pass to make the `Lint` workflow pass. No behaviour
changes.

- `ruff check --fix` auto-fixes: RUF100 (unused noqa) in
  `app/api/routes/conversation.py`, UP037 (quoted type annotations) x3
  in `app/cache/export_import.py`, UP012 (redundant byte-string call)
  in `app/cache/export_import.py`, I001 (import sorting) and F401
  (unused `ALLOWED_TIERS` import) in
  `tests/test_cache_export_import.py`.
- SIM108: converted `if`/`else` block to ternary in
  `app/api/routes/cache_api.py` (`tiers = ...` assignment).
- SIM105: replaced `try`/`except TypeError`/`pass` with
  `contextlib.suppress(TypeError)` in `app/cache/export_import.py`;
  added `import contextlib` to the stdlib import block.
- `ruff format` pass over 8 files:
  `app/api/routes/cache_api.py`, `app/api/routes/conversation.py`,
  `app/api/routes/traces_api.py`, `app/cache/cache_manager.py`,
  `app/cache/export_import.py`,
  `tests/test_cache_export_import.py`,
  `tests/test_cache_visibility.py`,
  `tests/test_streaming_middleware.py`.

Compatibility: no behaviour changes.

Commits: see `git log v0.21.0..v0.21.1`.

### 0.21.0 (MINOR) -- action-cache rename and v2 cache export envelope

User-facing rename of the second cache tier from "response cache" to
"action cache". Backwards compatible: legacy `response` continues to be
accepted everywhere the new `action` value is, the on-disk Chroma
collection literal is unchanged, and existing `cache.response.*`
setting keys are preserved so user-tuned thresholds survive the rename.

Highlights:

- API surface: every cache endpoint (`/api/admin/cache/stats`,
  `/entries`, `/flush`, `/export`, `/import`) accepts both `tier=action`
  (canonical) and `tier=response` (legacy alias). New responses emit
  `action` as the canonical tier name.
- Export envelopes are now stamped with `format_version: 2` and use the
  `tiers.action.entries` shape. `parse_envelope` still accepts a
  `format_version: 1` envelope with `tiers.response.entries` so
  exports created on 0.20.x remain importable on 0.21.0.
- Dashboard cache page (`container/app/dashboard/templates/cache.html`)
  shows the tier label "action" and uses the action-named query
  parameters when calling the API.
- `cache_manager.action_cache` exposed as an alias for
  `cache_manager.response_cache` so plugin and test code that already
  uses the new name compiles against the runtime.

Changes:

- Cache API entry-shape uses `tiers.action.entries` for new exports;
  importer accepts both `tiers.action` and `tiers.response`.
- Internal Chroma collection literal kept as `response_cache` (no
  data migration required).
- DB-stored settings keys remain `cache.response.threshold`,
  `cache.response.partial_threshold`, `cache.response.max_entries`.

Compatibility:

- Exports produced by 0.20.x (envelope `format_version: 1`,
  `tiers.response`) are still importable on 0.21.0.
- Settings keys are unchanged. No DB migration runs.
- The `tier=response` query/body/form value continues to work on
  every endpoint; clients can adopt `tier=action` at their own pace.

Commits: see `git log v0.20.1..v0.21.0`.

### 0.20.1 (PATCH) -- per-turn tracing for /ws/conversation

Fixes inflated `total_duration_ms` on the dashboard waterfall for HA
conversation turns delivered over the persistent
`/ws/conversation` socket. The `TracingMiddleware` previously
created a connection-level `SpanCollector` and called
`TraceSummaryRepository.update_duration` when the socket finally
closed, overwriting each per-turn duration with the entire
connection lifetime.

- `container/app/middleware/tracing.py`: `_handle_websocket` now
  bypasses connection-level trace creation for paths starting with
  `/ws/conversation`. It still exposes `state["source"] = "ha"` and
  a new `state["ws_per_turn"] = True` marker. All other WS paths
  keep the legacy per-connection trace + flush + `update_duration`
  behaviour.
- `container/app/api/routes/conversation.py`: `ws_conversation`
  now mints a fresh `trace_id` + `SpanCollector` + `root_span_id`
  per inbound message inside the receive loop, exposes them on
  `scope["state"]` for the duration of the turn, hands the
  collector to `_build_a2a_request`, and in `finally` appends a
  synthesised `ws_turn` root span and flushes the collector. The
  dead `connection_span` / `connection_root_span` fallback branch
  is removed and `FLOW-WS-SPAN-1` comments are updated to
  `FLOW-WS-TURN-1`.
- `container/tests/test_streaming_middleware.py`: flipped
  `test_tracing_middleware_populates_websocket_span` to assert the
  new bypass semantics for `/ws/conversation`; kept
  `test_tracing_middleware_ws_source_defaults_to_api` unchanged
  (legacy path); added `test_ws_conversation_mints_per_turn_trace`
  driving two synthetic turns through `ws_conversation` and
  asserting per-turn flushes and absence of any middleware-side
  `update_duration` call.

### 0.20.0 (MINOR) -- routing/response cache export and import

New admin endpoints and dashboard controls to back up and restore the
routing and response caches as a portable JSON envelope.

- New helper module ``container/app/cache/export_import.py`` with
  ``iter_export_chunks``, ``parse_envelope`` and ``import_envelope``.
- New endpoints ``GET /api/admin/cache/export`` (streams the envelope)
  and ``POST /api/admin/cache/import`` (multipart upload, ``mode``
  ``merge`` or ``replace``, ``tiers`` CSV, ``re_embed`` flag).
- Dashboard cache page gains a "Backup" card with export/import
  controls, mode and re-embed toggles.
- Per-entry validation drops malformed records and reports them in
  ``warnings``; envelope-level rejection returns HTTP 400.
- Imports run ``prepare_for_flush()`` to invalidate in-flight writes
  and a single ``_enforce_lru`` pass per affected tier afterwards.

### 0.19.3 (PATCH) -- scene_executor domain filtering

Closes the single HIGH-risk finding from the v0.19.2 executor domain-filter
audit. `scene_executor.py` was the only domain executor still selecting
`matches[0]` / `reranked[0]` from a domain-blind hybrid matcher and relying on
post-hoc `_validate_domain()` rejection. A request like "activate movie night"
in a home that also exposes `switch.movie_night_scene_control` could be
silently dropped by the post-hoc check instead of resolving to the in-domain
`scene.movie_night`.

- Threaded `filter_matches_by_domain(matches, _ALLOWED_DOMAINS)` into both
  `execute_scene_action()` and `_query_scene()` in
  `container/app/agents/scene_executor.py`, mirroring the
  filter -> rerank -> pick `[0]` shape from the six other patched executors.
- Kept the existing post-hoc `_validate_domain()` block as defence-in-depth.
- Added regression scenario `scene/activate_scene_disambiguation` with an
  injected wrong-domain look-alike (`switch.movie_night_scene_control`) to
  lock the behaviour in.

### 0.19.2 (PATCH) -- per-action domain filtering in domain executors

Fixes a security-critical cross-domain entity selection bug surfaced
by the new real-scenario suite. `EntityMatcher.match()` is domain-blind
and downstream domain executors blindly took `matches[0]`, so a
`camera_turn_on` action with the query "front door camera" could land
on `lock.front_door` (or vice versa for a lock action with a query
that ranked a camera/sensor higher).

- New helper `filter_matches_by_domain(matches, allowed_domains, *,
  fallback_to_unfiltered=False)` in
  `container/app/agents/action_executor.py` returns the order-preserving
  subset of matches whose `entity_id` belongs to `allowed_domains`
  (composes filter -> rerank -> pick `[0]`).
- Threaded the helper into the entity-resolution sites of
  `security_executor`, `climate_executor`, `media_executor`,
  `music_executor`, `timer_executor`, `automation_executor`, and the
  light/switch resolver in `action_executor` with per-action domain
  maps. Empty-after-filter falls into the existing not-found branch.
- Flipped the previously `xfail` scenario
  `security/camera_turn_on_camera` to passing and added regression
  scenarios `security/lock_front_door_disambiguation` and
  `security/turn_off_front_door_camera`.
- Added unit tests for `filter_matches_by_domain` in
  `container/tests/test_action_executor.py`.

### 0.19.1 (PATCH) -- real-pipeline scenario suite expansion

- Expanded real-scenario E2E suite to full 78-case coverage. Added 80
  new YAML scenarios under `container/tests/data/scenarios/` covering
  light (10), climate (10), media (7), music (10), scene (4),
  security (9), automation (4), timer (9), general (5), send (5), and
  orchestrator (7) flows. All run end-to-end against the production
  pipeline; 4 are marked `xfail: <reason>` for cases that require
  framework features not yet wired (cache replay, controlled dispatch
  latency, real DelayedTaskManager loop hookup, and ambiguous
  cross-domain entity ranking in `security_executor`).
- Framework additions in `container/tests/scenarios/`:
  - `Scenario.xfail` field (loader + dataclass) so individual YAMLs
    can carry a `pytest.mark.xfail(strict=False, reason=...)` marker.
  - `runner.run_scenario` seeds `send_device_mappings` rows from
    `preconditions.send_device_mappings` so send-agent scenarios
    resolve display names to HA service targets without DB hand-wiring.
- `container/tests/data/scenarios/README.md` Coverage section updated
  to reflect the full 78-scenario corpus.

### 0.19.0 (MINOR) -- real-pipeline scenario test suite

Adds a new YAML-driven end-to-end test framework that exercises the
production OrchestratorAgent pipeline against a curated HA snapshot,
deterministic LLM stubs, and an in-memory recording HA client. No
production source files are modified.

- New framework under `container/tests/scenarios/`:
  - `loader.py` parses snapshot and scenario YAML files.
  - `runner.py` builds the real pipeline (OrchestratorAgent + Dispatcher
    + InProcessTransport + AgentRegistry + EntityIndex + EntityMatcher
    + all ten production domain agents) backed by a `StubVectorStore`,
    a temporary aiosqlite DB seeded via `app.db.schema._seed_defaults`,
    and a reset `HomeContextProvider` singleton.
  - `recording_ha_client.py` implements the `HARestClient` surface
    used by the routable agents and records every `call_service`
    invocation while plausibly mutating in-memory state for downstream
    `expect_state` / `get_state` calls.
  - `deterministic_llm.py` provides a FIFO-per-`agent_id` reply stub
    that raises `LLMStubMissError` when an agent calls `complete(...)`
    without a queued reply.
  - `embedding_stub.py` produces deterministic 384-dim embeddings via
    BLAKE2b digests; the stub vector store uses token-overlap distance
    so candidate ranking is stable for the fixture corpus.
- New fixtures under `container/tests/data/`:
  - `ha_snapshots/home_default.json` (52 entities across light, switch,
    climate, weather, sensor, binary_sensor, media_player, scene, lock,
    alarm_control_panel, camera, cover, automation, timer,
    input_datetime, input_boolean, assist_satellite domains) plus
    `home_default.areas.json`, `home_default.devices.json`, and
    `home_default.config.json` for area/device/timezone wiring.
  - `scenarios/` containing 14 representative YAML scenarios covering
    light (3), climate (1), media (2), music (1), scene (1), security
    (2), automation (1), timer (1), general (1), and orchestrator (1).
- New parametrised pytest entry `container/tests/test_real_scenarios.py`
  marked with the new `real_scenarios` marker registered in
  `container/pyproject.toml`. New scenario YAML files added to
  `tests/data/scenarios/**` are picked up automatically.
- Includes scenario authoring docs under
  `container/tests/data/scenarios/README.md` (YAML schema cheat sheet,
  debugging guide for `LLMStubMissError`, coverage status).

Send-agent and orchestrator meta-action coverage (cancel_interaction,
multi-step composition) are intentionally deferred to a follow-up
because they need additional fixtures (notify-target seeding,
mid-turn cancellation contracts).

Validation:

- Real-scenario suite: `14 passed in 2.37s`
  via `python -m pytest tests/test_real_scenarios.py`.

### 0.18.39 (PATCH) -- dashboard auth expiry and HA integration UX fixes

Fixes a small cluster of reviewed dashboard and Home Assistant integration
defects without widening the scope into deferred hardening work.

- Added a shared dashboard-side fetch helper in
  `container/app/dashboard/templates/dashboard_base.html` and moved the
  reviewed long-lived dashboard pages onto it so `401` and `HX-Redirect`
  auth-expiry responses now force a full redirect back to
  `/dashboard/login` instead of leaving stale pages running.
- Made the dashboard agent editor in
  `container/app/dashboard/templates/agents.html` acknowledgment-driven:
  toggles, config saves, prompt writes, and MCP assignment changes now wait
  for backend success before treating the UI as saved, surface visible
  per-agent failures, and reload backend truth after rejected writes.
- Tightened agent editor backend write responses in
  `container/app/api/routes/dashboard_api.py` so failed config and prompt
  writes return structured JSON error details the dashboard can display.
- Refactored `custom_components/ha_agenthub/config_flow.py` so setup and
  options share the same health-payload validation, the API key field uses
  password-style selector semantics, and leaving the options API key blank
  now keeps the stored secret instead of re-exposing it in the form.
- Improved HA REST fallback messaging in
  `custom_components/ha_agenthub/conversation.py` to distinguish rejected
  API keys, backend/container errors, and unreachable-container failures
  with more actionable user-facing guidance.
- Applied small accessibility fixes across the touched dashboard pages:
  the mobile sidebar toggle now exposes `aria-expanded` and
  `aria-controls`, the send-devices add form uses explicit label/input
  associations and real form submission, and async feedback regions on the
  touched pages are marked as live regions.

Validation:

- Focused dashboard coverage: `39 passed, 1 warning in 6.77s`
  via `python -m pytest tests/test_dashboard.py -q`.
- Focused HA integration coverage: `47 passed in 3.95s`
  via `python -m pytest tests/test_ha_client.py -q`.

### 0.18.38 (PATCH) -- test suite runtime optimizations

Low-risk test-runtime improvements focused on keeping behaviour and coverage
unchanged while removing avoidable waits and setup overhead in the local and
CI pytest workflows.

- Deferred the local `litellm` import in
  `container/app/api/routes/admin.py::test_llm_provider` and
  `container/app/setup/routes.py::test_llm_endpoint` so unknown-provider and
  missing-key fast-fail paths return before paying the cold import cost.
- Shortened real timeout and retry waits in targeted tests by patching the
  existing timeout and retry-delay boundaries inside
  `container/tests/test_agents.py`, `container/tests/test_llm.py`, and
  `container/tests/test_mcp.py` instead of mocking away the timeout/retry
  mechanisms themselves.
- Stubbed `app.setup.routes.hash_password` only in setup/auth/CSRF route tests
  that validate form flow and repository invocation, while keeping the direct
  bcrypt coverage in `container/tests/test_security.py::TestHashing` fully real.
- Added a shared integration-app builder in `container/tests/conftest.py` and
  reused it across the API, setup, dashboard, CSRF, MCP, and security
  integration-style test modules without widening fixture scope, then reran the
  touched modules twice to confirm no order dependence.
- Expanded the README test section with the canonical serial command, a fast
  `-m "not integration"` inner-loop command, and the local `-n auto` command,
  while clarifying that `pytest-xdist` requires the dev dependencies to be
  installed in the current environment.

Validation:

- Full serial suite: `1273 passed, 3 skipped, 1 warning in 43.08s`
  (baseline before changes: `48.88s`, improvement: `5.80s`, about `11.9%`).
- Local `pytest-xdist` validation was not run in this environment because
  `python -m pip show pytest-xdist` reported `Package(s) not found: pytest-xdist`.

### 0.18.37 (PATCH) -- entity-name translation fix

Stops the LLM from translating localized room/device/scene/automation
names into English (or any other language) when generating condensed
tasks or domain actions. For example, a German user saying "schalte
das Licht im Keller ein" no longer ends up addressing a non-existent
"cellar light" entity; "Keller" is preserved verbatim and matches the
real ``light.keller``.

- ``container/app/prompts/orchestrator.txt`` and the eight actionable
  domain prompts (``light``, ``climate``, ``scene``, ``security``,
  ``timer``, ``media``, ``music``, ``automation``) now include a
  uniform ``CRITICAL -- ENTITY NAMES MUST NEVER BE TRANSLATED`` block
  inserted between the schema/format header and the few-shot examples.
  The block is universally English so it does not bias the LLM toward
  any particular target language.
- ``container/app/agents/actionable.py`` now PREPENDS a multi-line
  language directive to the loaded system prompt (instead of appending
  a single line) for every non-English request. Time/location context
  remains appended (it is data, not a constraint).
- No code-level behaviour change; prompt-only patch validated by live
  probes (German requests now resolve to the real entities) and by the
  full test suite (1273 passed, 3 skipped).

### 0.18.36 -- Phase 3 chunk 6 (P3-6, P3-9, P3-10, P3-11)

Final Phase-3 plan items. Backwards compatible; no schema or wire
changes. See ``docs/SubAgent/full_flow_plan.md`` Phase 3.

- **P3-6 (settings TTL cache)**: ``SettingsRepository.get_value`` now
  serves from an in-memory dict cache with a 60s TTL
  (``_SETTINGS_VALUE_CACHE_TTL_SEC`` in
  ``container/app/db/repository.py``). Cache hits skip the SQLite read
  entirely; misses (including absent keys, stored as a sentinel) are
  populated. ``SettingsRepository.set`` invalidates the affected key
  so subsequent reads observe the write. A test-only autouse fixture
  in ``container/tests/conftest.py`` clears the cache between tests
  to keep the per-test temporary databases isolated. Public API and
  return values are unchanged.
- **P3-9 (dispatch-path consolidation)**: Re-evaluated after P1-1
  iterations 1-3. The shared helpers (``_dispatch_single``,
  ``_handle_sequential_send``, ``_do_cache_lookup``, ``_classify``,
  ``_finalize_single_agent_response``, ``_create_trace``,
  ``_store_response_cache``) already cover all non-streaming-specific
  logic, and the streaming impl delegates multi-agent and
  sequential-send back to ``handle_task``. The remaining differences
  are the genuine streaming primitives (token relay and the
  filler/queue race) that P1-1 documented as a real architectural
  difference. P3-9 is therefore considered done by P1-1; this is
  recorded in the docstring of ``Orchestrator._run_pipeline``.
- **P3-10 (logging audit)**: Demoted 13 per-request hot-path
  ``logger.info`` calls in ``container/app/agents/orchestrator.py``
  to ``logger.debug`` (per-request "Routed to ...", "Stream routed
  ...", "Routing cache hit", "Classification LLM response",
  "Agent ... responded in ...", filler decision/timing/result/sent
  logs, cached-action visibility note, cache-replay fall-through).
  Real anomalies (timeouts, dispatch errors, mediation failures,
  cache-write failures) are kept at ``warning``. No log assertions
  in the test suite are affected.
- **P3-11 (magic numbers)**: Extracted module-level constants for
  the remaining timing literals: ``_FILLER_LLM_TIMEOUT_SEC = 3.0``
  in ``container/app/agents/filler.py``,
  ``_LLM_EMPTY_RESPONSE_RETRY_DELAY_SEC = 1.0`` in
  ``container/app/llm/client.py``,
  ``_OWNER_TASK_DISCONNECT_TIMEOUT_SEC = 5.0`` in
  ``container/app/mcp/client.py``,
  ``_ENTITY_SYNC_DEFAULT_INTERVAL_MIN = 30`` /
  ``_ENTITY_SYNC_DISABLED_RECHECK_SEC = 300`` in
  ``container/app/main.py`` and ``container/app/runtime_setup.py``,
  and ``_ENTITY_UPDATE_FLUSH_INTERVAL_SEC = 0.5`` in
  ``container/app/runtime_setup.py``. Notification-dispatcher
  delays were already extracted in earlier work.

### 0.18.35 -- Phase 3 chunk 5 (P3-1, P3-2, P3-3, P3-4, P3-5, P3-7, P3-8)

Seven low-risk Phase-3 plan items (all S effort, see
``docs/SubAgent/full_flow_plan.md``). Backwards compatible; default
behaviour for existing clients unchanged.

- **P3-1 (sanitized response flag)**: ``ConversationResponse`` and
  ``StreamToken`` (``container/app/models/conversation.py``) gained an
  optional ``sanitized: bool`` field defaulting to ``True``. The HA
  custom component (``custom_components/ha_agenthub/conversation.py``)
  now trusts the flag: ``_build_result`` accepts a ``sanitized`` kwarg
  and only re-runs ``_strip_markdown`` when the backend explicitly
  reports unsanitised text. Both REST and WS paths propagate the value
  (REST reads ``data.get("sanitized", False)``; WS tracks
  ``stream_sanitized`` across chunks and reads the final ``done``
  frame). The defensive ``_strip_markdown`` helper is preserved for
  legacy / older backends but is now a no-op for sanitized payloads.
- **P3-2 (auth handshake timeouts)**: introduced
  ``AUTH_HANDSHAKE_TIMEOUT = 10.0`` in
  ``container/app/ha_client/websocket.py`` and wrapped both
  ``ws.receive_json()`` calls inside ``connect()`` (auth-required and
  auth-result frames) in ``asyncio.wait_for``. Prevents an idle HA
  server from leaving the connect coroutine blocked indefinitely
  during the auth handshake.
- **P3-3 (vector-store reinit lock)**: ``ChromaVectorStore`` now holds
  a ``threading.Lock`` and ``_reinitialize_sync`` runs its full body
  inside the lock with a double-checked guard
  (``if self._client is not None and self._is_alive(): return``).
  Concurrent threads that all observed a dead client now produce
  exactly one new ``PersistentClient`` instead of racing.
- **P3-4 (response-cache flush ordering)**: added
  ``ResponseCache.prepare_for_flush()`` which delegates to
  ``self._state.invalidate()`` (mirrors ``RoutingCache``).
  ``ResponseCache.store()`` now snapshots
  ``self._state.current_generation()`` BEFORE doing work and skips the
  upsert via ``matches_generation`` after the flush gate, so a flush
  that lands mid-store no longer resurrects the cleared entry.
  ``CacheManager.flush()`` calls ``prepare_for_flush()`` for the
  ``response`` tier (and both caches when no tier is given) before
  the underlying delete.
- **P3-5 (state-waiter cleanup on disconnect)**: added
  ``WebSocketReset`` exception and ``_cancel_all_state_waiters(reason)``
  on the HA WebSocket client; ``_close_session`` snapshots and clears
  the waiter map and sets ``WebSocketReset`` on every pending future
  (under ``contextlib.suppress(InvalidStateError)``). Pending
  ``async_wait_for_state`` callers now wake immediately on reconnect
  instead of hanging until their per-call timeout.
- **P3-7 (known-agents memoisation)**: ``OrchestratorAgent`` now
  caches the ``registry.list_agents()`` result with a 5 s TTL
  (``_known_agents_cache``, ``_known_agents_ttl``).
  ``_load_reliability_config`` invalidates the memo so deliberate
  reconfigurations remain authoritative; setting the TTL to ``0``
  disables the cache for tests / stress paths.
- **P3-8 (notification dispatcher agent_id)**: the TTS
  ``llm.complete`` call in
  ``container/app/agents/notification_dispatcher.py`` now passes
  ``agent_id="notification-dispatcher"`` instead of the (incorrect)
  ``"orchestrator"``, restoring per-agent budgeting and metrics.

Tests: 14 new tests in
``container/tests/test_phase3_chunk5.py`` (``TestSanitizedFlagDefault``,
``TestVectorStoreReinitLock``, ``TestResponseCachePrepareForFlush``,
``TestKnownAgentsMemoization``, ``TestNotificationDispatcherAgentId``)
plus 3 new tests in
``container/tests/test_ha_websocket_waiters.py``
(``TestStateWaiterReconnectCleanup``).
Full suite: ``1268 passed, 3 skipped, 1 warning in 59.44s``
(was ``1251 passed, 3 skipped`` -- +17 new tests, zero regressions).

Deferred to chunk 6: P3-6 (settings in-memory cache, M),
P3-9 (multi-agent / sequential-send consolidation, M),
P3-10, P3-11.

### 0.18.34 -- Fix pre-existing test failures (WS close-error contract + tzdata)

- **Test fix (TestHAConversationWSCloseError)**: the four
  ``test_ha_client.py::TestHAConversationWSCloseError`` tests were
  drifting against the current ``_process_via_ws`` contract from
  0.18.27 onward. CLOSED/ERROR mid-stream and a JSON-decode/timeout
  failure are wrapped by the outer ``except`` clause as
  ``_WsDroppedAfterSendError`` (with the original ``aiohttp.ClientError``
  attached as ``__cause__``) so the conversation entity can suppress
  the duplicate REST fallback. ``done``-chunks containing ``error``
  are intentionally NOT raised -- they are logged and embedded in the
  speech result. Tests now assert this exact behaviour
  (``_WsDroppedAfterSendError`` with the cause message check for
  the first three; ``_build_result`` invocation with the embedded
  error string for the fourth).
- **Dev dependency (tzdata)**: added ``tzdata>=2024.1`` to
  ``container/requirements-dev.txt`` so
  ``test_recorder_history.py::TestSummarizeHistory::test_numeric_min_max``
  can resolve ``ZoneInfo("UTC")`` on Windows / Python 3.14 venvs that
  lack the system tz database. Installed locally for the verification
  run.
- Full suite: ``1251 passed, 3 skipped, 1 warning`` -- previously
  blocking 5 failures cleared.

### 0.18.33 -- Orchestrator pipeline shared finalize / classify-span helpers (P1-1 iter 3)

- **P1-1 iter 3 (FLOW-PIPE-1)**: extended the iter-2 dedup with two
  more behaviour-preserving helpers on ``OrchestratorAgent``:
  ``_pipeline_record_classify_span(...)`` populates the six base
  ``classify`` span metadata keys both pipeline impls always set,
  with an opt-in ``extended_metadata`` flag for the
  ``all_classifications`` key that only the non-streaming pipeline
  recorded; the streaming impl keeps the default ``False`` so its
  span payload is byte-identical to before.
  ``_finalize_single_agent_response(...)`` runs the shared
  ``return``-span block (mediation, voice-followup merge, response
  cache store, turn store, trace summary) for the single-agent and
  sequential-send paths. The helper accepts ``routed_to``,
  ``mediation_agent``, ``skip_mediation_on_error`` (NS=True for the
  agent-error guard, streaming=False) and ``skip_response_cache``
  (NS sequential-send=True to preserve the prior ``len==1``
  cache-store guard) so both callers keep their exact previous
  behaviour. Multi-agent NS finalization stays inline because the
  ``_merge_responses`` step has no streaming counterpart and runs
  before the mediation-skip check. Cancel-interaction handling was
  not extracted -- NS uses inline guards while streaming has a full
  early-return block, so a shared helper would not actually dedup.
  ``ORCHESTRATOR_LEGACY_PIPELINE`` rollback flag still works (verified
  with the full ``test_agents.py`` + ``test_orchestrator_pipeline.py``
  suite passing under the flag).

### 0.18.32 -- Orchestrator pipeline shared prelude helpers (P1-1 iter 2)

- **P1-1 iter 2 (FLOW-PIPE-1)**: extracted two behaviour-preserving
  helpers from ``OrchestratorAgent`` to remove the most drift-prone
  copy-paste between ``_handle_task_impl`` and
  ``_handle_task_stream_impl``:
  ``_pipeline_resolve_conversation_and_language(task)`` resolves
  ``conversation_id`` (with uuid fallback), the effective language
  via ``_resolve_language``, and prefetches conversation turns;
  ``_pipeline_try_response_cache_replay(...)`` performs the
  response-hit replay, opens the ``cache_fallthrough`` span on
  failure and emits the appropriate (stream / non-stream) info log.
  Both pipeline impls now call these helpers; classify, dispatch,
  filler, mediation and trace blocks were left intact because their
  span-metadata and control-flow shapes diverge enough that
  consolidating them would risk subtle behaviour changes. The
  ``ORCHESTRATOR_LEGACY_PIPELINE`` rollback flag remains live and
  the legacy impls remain reachable via that path.

### 0.18.31 -- Per-agent dispatch timeouts and parse_action schema validation

- **P2-2 (FLOW-TIMEOUT-1)**: orchestrator dispatch no longer applies a
  single 5s timeout to every sub-agent. ``AgentCard`` now carries an
  optional ``timeout_sec`` field; ``OrchestratorAgent`` resolves the
  effective timeout per agent_id with priority
  ``agent.dispatch_timeout.<agent_id>`` setting > AgentCard.timeout_sec
  > ``a2a.default_timeout`` (still 5s). The resolved value is capped at
  ``a2a.max_dispatch_timeout`` (default 60s) and cached per agent_id
  for the lifetime of the orchestrator instance so SettingsRepository
  is hit at most once per agent. ``general-agent`` and dynamic
  (custom plugin) agents now declare ``timeout_sec=30.0`` so MCP /
  web-search calls are no longer killed mid-call. New tests:
  ``tests/test_per_agent_timeout.py``.
- **P2-6 (FLOW-PARSE-1)**: ``action_executor.parse_action`` now
  validates every candidate JSON object against a Pydantic
  ``ActionPayload`` schema (``action`` non-empty string + an entity /
  entity_id, except for explicit aggregation actions like
  ``list_lights`` / ``list_timers``). When the JSON in a fence
  decodes but fails schema validation, the parser falls through to
  the next regex / inline scan instead of returning the bad payload.
  Backwards compatible: the public function still returns ``dict |
  None`` and accepts ``entity_id`` as a synonym for ``entity``. New
  tests added under ``tests/test_action_executor.py::TestParseAction``.

### 0.18.30 -- Orchestrator pipeline entry point and streaming mediation pass-through

- **P1-1 (FLOW-PIPE-1)**: introduced ``OrchestratorAgent._run_pipeline(task, *,
  streaming)`` as the unified pipeline entry point. The bodies of
  ``handle_task`` and ``handle_task_stream`` were renamed to
  ``_handle_task_impl`` and ``_handle_task_stream_impl``; the public
  methods are now thin wrappers that route through ``_run_pipeline``.
  The streaming token sequence, multi-agent merge, sequential-send
  filler timing, cache-hit short-circuits, cancel-interaction shortcut
  and every existing FLOW-XXX fix call site are preserved unchanged.
  Setting ``ORCHESTRATOR_LEGACY_PIPELINE=1`` bypasses the wrapper and
  calls the impls directly as an emergency rollback lever for any
  follow-up deep-dedup refactor.
- **P2-1 (FLOW-MED-8 update)**: streaming mediation no longer suppresses
  sub-agent tokens when ``personality.prompt`` is set. The user now
  sees the streamed tokens as they arrive; the terminal ``done`` chunk
  still carries ``mediated_speech`` so clients can replace the streamed
  text with the mediated rewrite at end-of-stream. Dispatch span
  metadata key renamed from ``mediation_suppressed_interim_tokens`` to
  ``mediation_streamed_interim_tokens``.

### 0.18.29 -- Full-flow plan: setup dedupe, WS trace, coalescing, reconnect cap, cache telemetry

- **P1-2 (FLOW-SETUP-1)**: extracted the setup-dependent init sequence shared by
  ``main.lifespan`` and ``runtime_setup.ensure_setup_runtime_initialized`` into
  a single idempotent helper ``_initialize_setup_dependent_services``.
  Post-wizard reloads and lifespan startup now run exactly one HA client,
  cache manager, presence detector, WS client, and alarm monitor instance.
- **P1-6 (FLOW-WS-SPAN-1)**: ``TracingMiddleware`` now handles WebSocket
  scopes. Every WS connection gets its own outer SpanCollector with the
  source derived from the path (``/ws/conversation`` → ``ha``, fallback
  ``api``), and per-message turn spans are nested as children. Removed the
  hardcoded ``source="ha"`` fallback in ``/api/conversation``.
- **P2-3 (FLOW-COALESCE-1)**: HA custom-component conversation entity now
  coalesces repeat user turns arriving within a 250ms window onto the
  existing in-flight bridge task instead of firing a duplicate bridge.
- **P2-4 (FLOW-RECONN-1)**: HA WebSocket client reconnect loop hard-caps
  attempts (``MAX_RECONNECT_ATTEMPTS = 10``) and pauses for
  ``RECONNECT_PAUSE_DURATION = 300s`` before resetting the counter, so
  sustained HA downtime no longer busy-loops exponential-backoff forever.
- **P2-5 (FLOW-TELEM-1)**: ``CacheManager.process`` only emits a
  ``cache.track_cache_event`` analytics row for real hits (``response_hit``,
  ``response_partial``, ``routing_hit``); misses no longer churn the
  analytics store. Similarity is propagated into the event payload.

### 0.18.27 -- Stream: handle_task exceptions vs transport error chunk

- **BaseAgent.handle_task_stream** (default): exceptions from ``handle_task`` are
  caught and turned into a normal ``TaskResult`` with ``INTERNAL`` error speech,
  so **InProcessTransport** does not emit the generic ``Agent error: {id}`` stream
  chunk (which set ``stream_error`` and confused the HA WebSocket client).
- **InProcessTransport.stream**: re-raises ``asyncio.CancelledError``; other failures
  include ``{agent_id}: {ExcType}: {message}`` in the diagnostic chunk.

### 0.18.26 -- Routing flush vs in-flight store race

- Admin cache flush now calls ``RoutingCache.prepare_for_flush()`` before
  deleting Chroma routing rows: pending hit-count buffers are cleared and
  in-flight ``store_routing`` worker threads skip ``upsert`` if a flush
  happened mid-write. Without this, a classify that finished just before
  flush could still repopulate the routing tier so the next request showed
  ``routing_hit`` immediately after “flush all”.

### 0.18.25 -- LLM whitespace retry for HA voice

- The shared ``llm.client`` now treats whitespace-only model output as empty,
  so it triggers the existing retry path instead of passing a blank response
  into the agent layer.
- ``general-agent`` keeps the response generation in the LLM path and only
  turns a still-empty result into a standard ``llm_empty_response`` agent error.

### 0.18.24 -- HA voice stream error fallback deduplication

- Streaming requests routed to ``general-agent`` now return the same canned
  user-facing fallback speech as the REST path when the agent fails before
  producing any text, instead of surfacing an empty WS error.
- This prevents the Home Assistant integration from retrying the same utterance
  via REST after a failed WS stream, which previously created duplicate traces
  and left voice requests without a final response.

### 0.18.23 -- Non-blocking entity index warm-up

- App startup no longer waits for the full HA entity index populate/sync before
  the dashboard becomes reachable; the index now warms in the background.
- Extended health reports entity index `building` / `syncing` as a warning
  state instead of making startup feel like a dead page.

### 0.18.22 -- In-process post-setup runtime bootstrap

- Completing the setup wizard now initializes the HA-dependent runtime in the
  running process instead of leaving `ha_client`, `entity_index`, and
  `cache_manager` uninitialized until container restart.
- Admin overview/health/cache/entity-index routes now attempt the same guarded
  bootstrap when setup is complete but the runtime is still stale.
- Dashboard overview health badges no longer throw Alpine errors before the
  health payload finishes loading.

### 0.18.21 -- Docker Compose GHCR tag

- **docker-compose.yml** default image tag **:main** (CI never published **:latest** before).
  Override with ``HA_AGENTHUB_TAG`` (e.g. ``latest`` after next main push).
- **docker-build** workflow also pushes **:latest** on pushes to **main**.

### 0.18.20 -- CI tests faster and more reliable

- **GitHub Actions** ``Tests`` job: timeout **30** minutes (was 15), ``pytest -n auto`` via
  **pytest-xdist**, quieter output (``-q --tb=short``).
- **conftest:** separate SQLite/Chroma/Fernet paths per **xdist worker** to avoid races.

### 0.18.19 -- Setup wizard test coverage

- Integration tests for **step 4** Groq/Ollama URL branches and **LLM test** endpoint
  for ``groq`` and ``ollama`` providers (full branch coverage in ``app/setup/routes.py``).

### 0.18.18 -- Parallel tool execution in ``complete_with_tools``

- When the LLM returns **multiple** ``tool_calls`` in a single assistant message, the
  container now runs ``tool_executor`` for each **concurrently** via ``asyncio.gather``
  (same ordering of tool messages for the follow-up LLM call).

### 0.18.17 -- Recorder history on light-agent and security-agent

- **Shared helper** ``app.ha_client.history_query.execute_recorder_history_query`` centralizes
  HA ``/api/history/period`` loading and speech summarization; **climate-agent** uses it too.
- **light-agent**: action ``query_entity_history`` for ``light`` / ``switch`` / ``sensor``
  (e.g. illuminance) entities.
- **security-agent**: action ``query_entity_history`` for locks, alarms, cameras, and security
  sensors. **query_security_state** resolution now uses the same area tie-break as other agents.

### Integration ``custom_components/ha_agenthub`` 0.5.8 -- Agent error in WS ``done`` chunk

- If the final WebSocket frame has ``done: true`` and an ``error`` field (agent/stream
  error from the container), treat it as a **completed** turn with user-facing text
  (``mediated_speech`` or a short fallback), not as a transport failure. Avoids
  incorrectly showing the “connection dropped” message from 0.5.7.

### Integration ``custom_components/ha_agenthub`` 0.5.7 -- No REST after WebSocket payload sent

- If the WebSocket **already sent** the JSON payload and the stream then fails
  (timeout, close, bad frame, etc.), the bridge **no longer** falls back to
  REST. That fallback duplicated work on the container and produced misleading
  ``routing_hit`` / “two prior messages” traces. The user gets a short spoken
  notice instead; use REST only when the WS path fails **before** ``send_json``.

### Integration ``custom_components/ha_agenthub`` 0.5.6 -- Duplicate request coalescing

- Conversation bridge **coalesces** parallel ``_async_handle_message`` calls that
  share the same ``conversation_id`` and user text into **one** WebSocket/REST
  round-trip (in-flight task shared). Mitigates duplicate container traces when
  Home Assistant invokes the same turn twice in quick succession or when WS and
  REST paths would otherwise overlap.

### Integration ``custom_components/ha_agenthub`` 0.5.5 -- Cancel via container LLM

- **Dismiss / nevermind** is decided by the **orchestrator classification LLM**, which
  can route to virtual agent ``cancel-interaction`` for varied phrasing (not only fixed
  keywords). The container returns a short acknowledgement without domain-agent dispatch.
- The HA integration **always forwards** the user text to the container (no local
  phrase shortcut).

### 0.18.16 -- Voice follow-up without ``area_id`` (Companion app)

- When Assist comes from a **phone** there is often **no ``area_id``**,
  but Home Assistant still supplies ``device_id`` (device registry).
  Voice follow-up now runs ``assist_pipeline.run`` with that registry
  id when ``area_id`` is missing or no satellite matches the area.

### 0.18.15 -- Voice follow-up from orchestrator (HA satellite)

- **TaskResult.voice_followup:** Specialized agents can set
  ``voice_followup=True`` so that after TTS the container calls
  ``assist_pipeline.run`` on the origin satellite (same mechanism as
  timer notifications). When ``TaskContext.source`` is ``ha``, requires
  ``area_id`` and/or ``device_id``.
- **Organic follow-up (optional):** Settings
  ``orchestrator.organic_followup_enabled`` (default false) and
  ``orchestrator.organic_followup_probability`` (default ``0.08``)
  occasionally append a short closing question and open the mic.
- **Delay:** ``orchestrator.voice_followup_delay`` (seconds) overrides
  the notification profile's ``tts_to_listen_delay`` when set.
- **API:** ``ConversationResponse`` / ``StreamToken`` include
  ``voice_followup`` for clients.

### 0.18.14 -- CI: Ruff clean + stable tests

- **Lint:** ``ruff check`` / ``ruff format`` pass on ``container/`` (import
  sort, typing, SIM/B/N rules, explicit re-exports, ``raise ... from None``
  where required, plugin hook bodies, test fixes).
- **Tests:** Language-detection cases mock ``langdetect`` via ``sys.modules``
  so they do not depend on the library being installed or on detector
  randomness; embedding init test mocks ``_get_local_model`` so CI does not
  need to load ``sentence_transformers`` weights.

### 0.18.13 -- HA integration display name (i18n)

- Root-level **title** in ``strings.json`` and translations so the
  integration picker and UI use **HA-AgentHub** consistently.
  ``INTEGRATION_TITLE`` in ``const.py``; config flow entry title uses
  the same constant.
- Custom component **manifest 0.5.3**.

### 0.18.12 -- GitHub owner ``mainzerp``

- Canonical repository: **https://github.com/mainzerp/ha-agenthub**.
  README, deployment, HACS, ``manifest.json`` **documentation** /
  **issue_tracker**, and **codeowners** updated to ``mainzerp``.
- Custom component **manifest 0.5.2**.

### 0.18.11 -- Git repository URL (documentation)

- Public GitHub repository name is **ha-agenthub** (was ``agent-assist``).
  README, deployment docs, HACS clone URL, and ``manifest.json``
  **documentation** / **issue_tracker** now point to the ha-agenthub
  repo (superseded by **0.18.12** for owner ``mainzerp``). Docker
  service name, volume, and image labels stay **agent-assist**
  (unchanged).
- Custom component **manifest 0.5.1** (documentation URLs only).

### 0.18.10 -- HA integration domain ``ha_agenthub`` (breaking for HA)

- Home Assistant integration **folder** is now
  ``custom_components/ha_agenthub/`` and ``manifest.json`` **domain**
  is ``ha_agenthub`` (was ``agent_assist``). **Remove** the old
  ``custom_components/agent_assist`` tree and any stale config entry,
  then add **HA-AgentHub** again in HA.
- Entity registry migration in ``conversation.py`` still rewrites
  legacy ``agent_assist`` conversation entities to the new entry id
  when possible.
- Config flow classes renamed to ``HaAgentHubConfigFlow`` /
  ``HaAgentHubOptionsFlow``; conversation entity class to
  ``HaAgentHubConversationEntity``.
- Custom component **manifest version 0.5.0**.

### 0.18.9 -- HA-AgentHub naming (HA integration + UI copy)

- Home Assistant custom component **display name** is now
  **HA-AgentHub** (``manifest.json``, HACS, ``strings.json``,
  ``en.json``, new ``de.json``, config flow entry title, device
  registry manufacturer/model). **Superseded by 0.18.10** for domain
  and folder rename.
- Custom component manifest **0.4.2** (historical).
- Dashboard settings copy, send-agent default notification title, and
  docs/README troubleshooting strings updated to match.

### 0.18.8 -- Container API Key in Dashboard Settings

- **Settings - Communication** includes **Home Assistant integration
  (HA to Agent Hub)**: status (key saved / not configured), masked
  suffix of the stored key, optional **custom key** save (PUT), and
  **Generate new key** (POST) with confirmation. The new key is shown
  once with copy-to-clipboard until dismissed.
- Admin API: ``GET /api/admin/container-api-key``,
  ``POST /api/admin/container-api-key/rotate``,
  ``PUT /api/admin/container-api-key`` (body: ``{"api_key": "..."}``,
  min 16 characters).
- Tests in ``tests/test_api.py`` (``TestAdminSettingsEndpoints``).

### 0.18.7 -- Home Assistant Connection in Dashboard Settings

- The **Communication** settings page now includes a **Home
  Assistant** block: base URL, long-lived access token (optional on
  save to keep the existing encrypted token), **Save** and **Test
  connection** actions. Saving persists ``ha_url`` to the settings
  table and optionally replaces the token via the existing secrets
  store, then calls ``HARestClient.reload()`` and
  ``HAWebSocketClient.drop_connection()`` so the running container
  picks up new credentials without a restart.
- New admin API: ``GET/PUT /api/admin/ha-connection`` and
  ``POST /api/admin/ha-connection/test`` (same probe as the setup
  wizard: ``GET {url}/api/`` with bearer auth).
- ``HAWebSocketClient.drop_connection()`` closes the socket while
  leaving ``run()`` alive so the reconnect loop uses fresh URL/token
  from the DB.
- Default seed adds ``ha_url`` (empty) for installs that open the
  dashboard before the setup wizard completes.
- Tests in ``tests/test_api.py`` (``TestAdminSettingsEndpoints``)
  cover the new endpoints.

### 0.18.6 -- Request Origin + Spatial Context End-to-End

- Adds end-to-end awareness of *where* a request originates: the
  chat UI, an HA conversation call from a voice satellite, or a
  raw API client. ``TaskContext`` grows a ``source`` literal
  (``"ha" | "chat" | "api"``) plus optional ``device_name`` /
  ``area_name`` alongside the existing ``device_id`` / ``area_id``.
  ``ConversationRequest`` accepts the same display names so the
  HA integration can ship them without a container-side registry
  lookup. The ``TracingMiddleware`` already derived ``source`` from
  the URL path; ``conversation.py`` and ``dashboard_api.py`` now
  forward that value into the ``TaskContext``.
- The HA custom component (``custom_components/ha_agenthub/
  conversation.py``) resolves the originating satellite's
  ``device_name`` via the device registry and its ``area_name`` via
  the area registry. Both flow through WS and REST fallbacks via a
  shared ``_resolve_origin_context`` helper. Dashboard chat is
  explicitly marked ``source="chat"`` with no spatial anchor, so
  agents don't silently pin a chat query to a previous satellite's
  area.
- Area-aware entity resolution: a new ``rerank_matches_by_area``
  helper in ``action_executor.py`` re-orders the hybrid matcher's
  candidate list so a same-area entity wins when it scores within
  0.05 of the top result. ``_select_deterministic_candidate`` gains
  a ``preferred_area_id`` tie-breaker applied to exact-name /
  exact-area matches. Light, climate and scene agents feed the
  satellite's ``area_id`` into their executors; ``ActionableAgent``
  now sets ``_current_task_context`` on every subclass so any
  executor can reach it without plumbing extra kwargs. Timer agent's
  historical override is removed in favour of the shared hook.
- Trace persistence: migration 16 adds ``device_id``, ``area_id``,
  ``device_name``, ``area_name`` columns to ``trace_summary`` and the
  insert path writes them for both live dispatch and response-cache
  replay. ``traces_api`` surfaces them, and the traces UI renders
  dedicated "Satellite" / "Area" tiles next to the existing Source
  badge (with the raw IDs in tooltips for debugging).
- New ``tests/test_area_tiebreaker.py`` (15 tests) covers the
  reranker edge cases (no area, single match, already-same-area,
  near-tie win, far-behind loss, no mutation of input), the
  deterministic tie-breaker, and the new ``TaskContext`` /
  ``ConversationRequest`` fields including source literal validation.

### 0.18.5 -- Unified Verified Service Call Across All Domain Executors

- Introduces a shared ``call_service_with_verification`` helper in
  ``app/agents/action_executor.py`` that centralises the pattern:
  register a WS ``state_changed`` waiter via ``ha_client.expect_state``
  **before** the REST call, invoke the service, then merge the REST
  reply with the observer's evidence into a uniform result dict
  (``success`` / ``entity_id`` / ``action`` / ``state`` / ``source``).
  The helper gracefully degrades when ``expect_state`` is not a real
  async context manager (legacy tests / bare ``AsyncMock``) so it can
  be adopted everywhere without breaking existing harnesses.
- ``build_verified_speech`` is the matching speech builder: it prefers
  the verified ``expected_state`` over the observed state, falls back
  to an action-specific intent phrase (``triggered``, ``snoozed``,
  ``activated`` …) when no target state is known, and never echoes a
  stale observation that contradicts the commanded intent.
- All seven domain executors now use both helpers instead of the old
  ``asyncio.sleep(0.3) + get_state`` pattern: **climate, media,
  music, security, scene, automation, timer**. Each executor declares
  a domain-local ``_EXPECTED_STATE_BY_ACTION`` map and ``_ACTION_PHRASES``
  table so that e.g. ``trigger_automation`` says "triggered" rather
  than a misleading "is now on" based on a coincidentally matching
  pre-existing state. ``timer_executor._snooze_timer`` routes its
  internal cancel/start pair through the same helper.
- ``orchestrator._execute_cached_action`` was migrated to the shared
  helper too, so cached replay and live dispatch share identical
  verification semantics.
- New ``tests/test_domain_executors_verify.py`` covers the helper
  directly (non-empty REST, empty REST + WS observer, exceptions,
  unsupported async-CM fallback) and adds one per-domain integration
  test for the "empty REST + observer confirms" scenario. Legacy
  fixtures in ``tests/test_action_executor.py`` and
  ``tests/test_agents.py`` were updated to share the new
  ``attach_expect_state_shim`` helper in ``tests/helpers.py``.

### 0.18.4 -- Cached Action Replay Uses WS State Verification

- `_execute_cached_action` now registers a WebSocket state waiter via
  `ha_client.expect_state` **before** calling the HA service, mirroring
  the live `action_executor.execute_action` path. Previously, when
  HA's REST `call_service` returned an empty list (the normal case for
  async-bus aktors like KNX/ABB/Zigbee2MQTT, where `state_changed`
  fires *after* the REST response), the replay was treated as failure
  and fell through to live dispatch. Result: the response-cache fast
  path was dead for exactly the hardware that needs it most.
- Empty REST response is now resolved against the WS observer: if the
  observed state matches the action's expected target (`turn_on` ->
  `on`, `turn_off` -> `off`, etc.), the replay is confirmed. `toggle`
  and other untargeted actions accept any observed change.
- Targeted actions with a mismatched observed state fall through to
  live dispatch so the user gets a truthful response instead of a
  stale confirmation.
- Covered by new tests in ``tests/test_agents.py ::
  TestExecuteCachedActionVerification`` (8 tests): non-empty REST
  authoritative, empty REST + observer confirms, empty REST + observer
  mismatch falls through, no observer evidence falls through, `None`
  call_result, toggle with / without observer, malformed cached actions.

### 0.18.3 -- Response Cache Priority + Read-Only Guard

- `CacheManager._process_inner` now checks the **response cache
  first**, then the routing cache. The previous routing-first order
  silently shadowed every response-cache entry, because the routing
  threshold (0.92) is below the response threshold (0.95): once a
  query had a routing entry, repeated hits never reached the response
  tier. Consequence: cached action replay + rewrite-agent variation
  were effectively dead code for repeated action queries -- every
  "schalte keller ein" re-ran the light-agent LLM turn.
- Response-cache hits only short-circuit when the entry carries a
  ``cached_action``. State queries (no replayable action) fall
  through to the routing tier so the agent recomputes against live
  HA state -- no more stale "Es sind 21 Grad" from a snapshot taken
  hours ago.
- ``response_partial`` still never short-circuits; it now surfaces
  only when the routing cache also misses, as a diagnostic signal
  for downstream consumers.
- Covered by new tests in ``tests/test_cache.py``:
  ``test_response_hit_with_cached_action_shadows_routing``,
  ``test_response_hit_without_cached_action_falls_through_to_routing``,
  ``test_response_partial_surfaces_only_when_routing_misses``.
  Pre-existing ``_process_inner`` tests were updated for the new
  query order (response first, routing second).

### 0.18.2 -- Entity Match Preview in Admin UI

- New **Entity Match Preview** card on `Entity Index` admin page:
  input a query and optional agent id, see exactly what an agent
  would receive -- the deterministic light-resolver result
  (`entity_id`, `friendly_name`, `resolution_path`, domain gate
  status) and the hybrid matcher's top-N candidates with per-signal
  scores (alias / embedding / levenshtein / jaro_winkler / phonetic).
- New endpoint `GET /api/admin/entity-index/match-preview?q=...&agent_id=...`
  runs both resolution paths and returns a visibility summary
  (rules and visible-entity count for the chosen agent) so operators
  can pinpoint why a query fails (visibility rule vs. matcher score
  vs. domain gate).
- Covered by `tests/test_entity_index_match_preview.py` (4 tests:
  happy path, empty query, domain gate reject, uninitialized index).

### 0.18.1 -- Post-Action State Verification via WebSocket

- Light/switch action verification now uses the live HA WebSocket
  stream (`state_changed` waiter registered **before** the service
  call) and falls back to short REST polling when the WS client is
  disconnected. The fixed `asyncio.sleep(0.3)` + single `get_state`
  is gone; stale post-action reads no longer contradict the user's
  intent on slow bus aktors (KNX/ABB).
- `HARestClient.expect_state(...)` async context manager centralises
  the verification and is wired to the running `HAWebSocketClient`
  from `main.py` at startup.
- `HAWebSocketClient.register_state_waiter` / `cancel_state_waiter`
  expose the per-entity waiter API used by `expect_state`.
- Intent-first speech in `execute_action`: when the verified state
  is unavailable or does not match the intent, the response reports
  the action ("turned off Keller") instead of asserting a stale
  observed state ("is now on").
- HA's synchronous `call_service` response is consulted as an
  authoritative source when it contains the target entity, before
  falling back to the WS/polling observer.
- New settings with sensible defaults:
  - `state_verify.ws_timeout_sec` (default `"1.5"`)
  - `state_verify.poll_interval_sec` (default `"0.25"`)
  - `state_verify.poll_max_sec` (default `"1.0"`)

### 0.18.0 -- Request-Flow Bug Fixes

- Cached action replay re-checks entity visibility (closes
  permission-bypass on cache hits).
- Cached HA service responses with empty body are treated as
  replay failures and fall through to live dispatch.
- Sequential-send no longer pipes timeout / error strings to the
  send-agent's content slot.
- `_dispatch_single` returns a structured error + canned message
  when the fallback agent itself errors (no more empty speech).
- Detected language is propagated to agents on the non-streaming
  dispatch path (REST + multi-agent + sequential-send).
- Multi-agent dispatch detects canned-error tuples as failures so
  "(Note: X could not be reached.)" works as designed.
- Timeout-fallback dispatch is attributed to the actual fallback
  agent in spans and request analytics.
- Routing and response cache keys now include language. ChromaDB
  cache collections are purged on first start of 0.18.0; SQLite
  schema is unchanged. The vector store under
  `container/data/vector/` is regenerated on demand.
- `notification_dispatcher` resolves a real HA device id for
  `assist_pipeline.run` and uses `EntityIndex.area` to find
  `assist_satellite` entities.
- HA component's filler TTS uses the correct `tts.speak` schema
  (`entity_id` = TTS engine, `media_player_entity_id` = speaker).
- HA component holds the WS lock across `_ensure_connected` and
  send to avoid spurious REST fallbacks.
- `HARestClient.reload()` rebuilds the underlying httpx client
  when `ha_url` changes after the setup wizard.
- Routing / response cache mutations and `home_context_provider`
  refreshes are de-duplicated under explicit locks.
- Filler skip-on-fast-agent uses an atomic queue probe.
- Sanitizer corpus shared between container and HA copy.
- General-agent / informational responses no longer enter the
  response cache by default.
- Conversation-turn store hydrates from the DB on miss.
- Streaming pipeline collapses to a single mediated chunk when
  personality is enabled.
- `SpanCollector` source is required at construction.
- `parse_action` accepts unlabelled markdown fences.
- `[SEQ]` prefix stripping tolerates leading whitespace.

### 0.17.0 -- Security & Reliability Hardening

- Setup wizard POST routes now require an admin session once setup is
  complete (closes pre-auth admin-takeover hole).
- CSRF tokens enforced on dashboard login and setup form POSTs.
- WebSocket auth no longer accepts query-string tokens.
- Session signing key is HKDF-derived with domain separation.
- Replaced shared aiosqlite read connection with per-call connections;
  fixes cross-task cursor corruption under load.
- Pure ASGI SetupRedirectMiddleware restores SSE/streaming first-byte
  latency.
- MCP client lifecycle moved to per-server owner task; clean shutdown.
- Orchestrator fallback dispatch now respects default_timeout.
- Span collector preserved across fallback dispatch (no more trace
  holes on retry).
- HA WebSocket: heartbeat=15, _ws_lock for reconnect serialization,
  and idle-detection added.
- Background tasks tracked in a module-level set to prevent GC drops.
- EntityIndex.list_entries pre-filters by domain in ChromaDB.
- Plugin imports run under a 10s timeout.
- Sanitization preserves ZWJ/ZWNJ for non-Latin scripts.
- Action JSON parser uses raw_decode (no false-positive brace
  matching).

Note: existing admin sessions are invalidated by the SEC-6 signing-key
change; users must log in again after upgrade.

### 0.16.1 -- Deterministic Light Resolution Fix

- **Deterministic light lookup before hybrid scoring**: `action_executor` now resolves exact entity IDs, exact friendly names, and benign variants such as `Keller light` before falling back to the hybrid matcher. This fixes false negatives for entities like `light.keller` with friendly name `Keller`.
- **Deterministic ambiguity handling**: Room/area fallback now only auto-selects when it resolves to one clear light candidate; otherwise it returns a clarification response instead of guessing.
- **Consistent entity ingestion paths**: Startup sync, admin refresh, and `state_changed` indexing now share the same HA-state parsing helper so direct matches are less likely to fail because one refresh path indexed different metadata.
- **Regression coverage**: Added tests for `Keller`, `Keller light`, direct entity-id resolution, refreshed-index behavior, deterministic read-path resolution, and shared state-ingestion helpers.

### 0.16.0 -- Location/Time Context, Weather Support, Embedding Warmup

- **Location/Time context for all agents**: All agents now receive city, timezone, and local datetime in their system prompts. Fetched from HA `/api/config` endpoint and cached (1h TTL) via `HomeContextProvider`. Manual override via `home.timezone` and `home.location_name` DB settings. Agents can answer time/location questions directly.
- **Climate agent weather support**: Climate agent can now query HA `weather.*` entities for current conditions (`query_weather`) and multi-day forecasts (`query_weather_forecast`). Weather questions are routed to climate-agent instead of general-agent. Uses HA-native weather data from the user's own integrations.
- **Embedding model preload at startup**: Local embedding model (`all-MiniLM-L6-v2`) is now loaded eagerly during `EmbeddingEngine.initialize()` instead of lazy-loading on first query, eliminating the ~7s cold-start penalty on the first cache lookup after container restart.
- **HA service response support**: `HARestClient.call_service()` now supports `return_response=True` parameter for HA response services like `weather.get_forecasts`.

### 0.15.0 -- Embedding Optimization

- **Startup: sync instead of populate**: When ChromaDB already has entity data from a previous run, startup now uses `sync()` (smart diff) instead of `populate()` (full re-embed), reducing startup embeddings from hundreds/thousands to only genuinely changed entities.
- **state_changed: skip unchanged embeddings**: `batch_add()` now compares incoming entity embedding text and metadata against what is in ChromaDB before upserting. Entities with unchanged embedding text are either skipped entirely or get metadata-only updates (no re-embedding).
- **Cache flush: metadata-only updates**: `RoutingCache._flush_pending_updates()` and `ResponseCache._flush_pending_updates()` now use `update_metadata()` instead of `upsert()` with documents, preventing re-embedding when only hit_count/last_accessed metadata changes.
- **VectorStore.update_metadata()**: New method that calls ChromaDB `update()` with only metadata, skipping the embedding function entirely.

### 0.14.5 -- Filler Safety + WS Keepalive

- **Always include mediated_speech in done chunks**: All three streaming done-chunk paths (streaming, sequential-send, multi-agent) now unconditionally include `mediated_speech`, ensuring consumers always have a clean final text override regardless of whether mediation changed the text or filler was sent.
- **WS heartbeat**: Added `heartbeat=15` to `ws_connect()` so aiohttp sends pings every 15s and marks the connection closed if no pong is received, reducing stale-connection detection from minutes to ~30s.
- **Idle connection verification**: `_ensure_connected()` now tracks `_ws_last_active` and sends a ping to verify the connection if idle for >60s before allowing a request to use it.
- **Immediate reconnect after REST fallback**: After a WS failure triggers REST fallback, a background reconnect task is scheduled immediately so the next request can use WS instead of waiting up to 30s for the reconnect loop.
- **Race protection**: `_connect_ws()` now acquires `_ws_lock` to prevent concurrent connection attempts from `_reconnect_loop()` and `_ensure_connected()`.

### 0.14.4 -- Cache Dedup + Hit Counter Flush

- **Deterministic cache IDs**: `RoutingCache.store()` and `ResponseCache.store()` now use `sha256(query_text)[:16]` instead of `uuid4()` for entry IDs, so upsert overwrites existing entries for the same query instead of creating duplicates.
- **Flush on store**: Both caches now flush pending hit-count updates inside `store()` to prevent buffered updates from being lost.
- **Reduced flush interval**: Lowered `_flush_interval` from 50 to 5 so low-traffic instances flush hit counts more frequently.
- **Shutdown flush**: Added `flush_pending()` public method to both caches and `CacheManager`; called during container shutdown to persist any remaining buffered hit counts.

### 0.14.3 -- Routing Cache Priority + Purge Widening

- **Routing cache checked first**: Reordered `_process_inner()` in `CacheManager` to check the routing cache before the response cache. The routing cache is cheaper (agent_id + condensed_task only) and prevents stale response entries from preempting valid routing hits.
- **Failed replay resets cache_result**: After a `response_hit` replay fails in `handle_task()` / `handle_task_stream()`, `cache_result` is now reset to `None` so `_classify()` can re-check the routing cache instead of skipping it.
- **Wider read-only purge**: `purge_readonly_entries()` now also removes entries whose `cached_action` contains a read-only service call (`query_*`, `list_*`), not just entries with empty `cached_action`.

### 0.14.2 -- Cache Fall-Through Span + Stale Purge

- **Cache fall-through span**: Added `cache_fallthrough` span in both `handle_task()` and `handle_task_stream()` to mark the gap when a cached action replay fails and the request falls through to live classify+dispatch. Visible in the Gantt trace timeline (rose color).
- **Startup purge of stale read-only cache entries**: Added `purge_readonly_entries()` to `ResponseCache` and `CacheManager`. A one-time background task at startup removes response cache entries with `cached_action == ""` (read-only responses like sensor queries that hold stale state values).

### 0.14.1 -- LLM Entity Particle Stripping

- **Prompt-level entity extraction fix**: Updated all 8 domain agent prompts (climate, light, scene, media, security, music, automation, timer) and the orchestrator prompt to instruct the LLM to strip grammatical particles (prepositions, articles, cases) from entity names in any language. Fixes degraded entity matching when users include particles like "im", "in the", "dans la", "en la" in their queries (e.g. "im gaestezimmer" -> "gaestezimmer").

### 0.14.0 -- Sensor Cache Fix

- **Prevent caching of read-only responses**: Added `cacheable` field (default `True`) to `ActionExecuted` model. All 8 executors now return `cacheable=False` for read-only actions (`_query_*`, `_list_*`). `_store_response_cache()` checks this flag and skips storage when `False`, preventing stale sensor/status data from being served from cache.
- **Fall-through on failed cached action replay**: When `_handle_response_cache_hit()` detects that a cached action existed but replay returned `None` (failed), it returns `None` so `handle_task()` and `handle_task_stream()` fall through to full classify+dispatch instead of returning stale text.

### 0.13.9 -- Digraph Embedding Search Fix

- **Digraph-to-umlaut dual embedding search**: When the query contains German digraphs (ae/oe/ue), `EntityMatcher.match()` now runs a second embedding search with the digraph-converted umlaut form (e.g., "gaestezimmer" -> "gastezimmer") and merges both result sets, de-duplicating by entity_id and keeping the better embedding score. This fixes the gatekeeper problem where the embedding model returned completely wrong entities for digraph queries.

### 0.13.8 -- Umlaut Containment Bonus Fix

- **Umlaut normalization**: Containment bonus in `EntityMatcher.match()` now normalizes both query and friendly_name via `_normalize_for_containment()` before substring check, handling German umlaut digraphs (ae/oe/ue) and Unicode diacritics so that queries like "gaestezimmer" correctly match friendly names like "Gastezimmer Temperatur"

### 0.13.7 -- Event Loop Blocking Fix

- **Async entity index**: Added async wrappers (`add_async`, `remove_async`, `populate_async`, `sync_async`, `search_async`) to `EntityIndex` that offload synchronous ChromaDB/embedding operations to thread pool via `run_in_executor()`
- **Batched state updates**: `on_state_changed` WebSocket callback now queues entity updates into an async queue, flushed every 0.5s in a single batch upsert, preventing event loop starvation on initial HA connect
- **Non-blocking startup**: `entity_index.populate()` during startup now runs in executor, allowing other async tasks to proceed
- **Non-blocking periodic sync**: `_periodic_entity_sync()` now uses `sync_async()` to avoid blocking the event loop every 30 minutes
- **Non-blocking search**: `EmbeddingSignal.score()` and `timer_executor` entity search calls now use `search_async()`, preventing event loop blocking during HTTP request handling

### 0.13.6 -- Entity Matcher Containment Bonus

- **Containment bonus**: Entity matcher now adds a 0.3 score bonus (capped at 1.0) when the normalized query is a substring of the candidate's friendly name, improving partial entity name matches

### 0.13.5 -- Entity ID Validation & Visibility Ordering Fix

- **Pydantic ValidationError fix**: `ActionExecuted.entity_id` now uses `result.get("entity_id") or ""` to handle `None` values that triggered Pydantic validation errors
- **Visibility filtering before top-N**: `EntityMatcher.match()` now applies agent visibility rules before the top-N cutoff, ensuring filtered-out entities do not consume result slots

### 0.13.4 -- Span Visibility Fix for Read-Action Paths

Threaded `span_collector` into all read-action code paths so `entity_match` spans appear in the trace timeline for status queries (not just write actions):

- **Read-action span threading**: Added `span_collector=None` parameter to all 8 `_handle_*_read_action()` and `_query_*()` functions; wrapped `entity_matcher.match()` calls with `_optional_span` in `action_executor`, `climate_executor`, `media_executor`, `music_executor`, `scene_executor`, `security_executor`, `timer_executor`, and `automation_executor`
- **Migration 14**: Seeded `device_class_include` rules for climate-agent (9 classes), security-agent (11 classes), and light-agent (1 class), plus `domain_include` sensor/binary_sensor rules using `INSERT OR IGNORE` for idempotency
- **Gantt colorMap**: Added `entity_match` to the vis.js Gantt chart color map (purple `#a855f7`)

### 0.13.3 -- Entity Match Span Tracing

Added `entity_match` spans to all 8 executor modules so entity resolution timing, scores, and signal breakdowns appear in the span timeline visualization:

- **Shared span utilities**: Moved `_NoOpSpan` and `_optional_span` from `orchestrator.py` to `app/analytics/tracer.py` for reuse across all executors
- **Span instrumentation**: Wrapped `entity_matcher.match()` calls in all 8 executors (`action`, `climate`, `media`, `music`, `scene`, `security`, `automation`, `timer`) with `entity_match` spans recording query, match_count, top_entity_id, top_friendly_name, top_score, and signal_scores
- **span_collector threading**: Added `span_collector=None` parameter through `ActionableAgent._do_execute()`, all 8 agent wrappers, all 8 executor main functions, and 5 timer helper functions (`_snooze_timer`, `_start_timer_with_notification`, `_sleep_timer`, `_create_reminder`, `_create_recurring_reminder`)

New tests: test_action_executor (3) = 3 new tests (entity_match span recorded, no-match span, backward compatibility)

### 0.13.2 -- Agent Domain Access Hardening

Removed unfiltered `entity_index.search()` fallback from all 8 executor modules and added per-executor domain validation:

- **Fallback removal**: Removed 17 occurrences of unfiltered `entity_index.search()` fallback across `action_executor`, `climate_executor`, `media_executor`, `music_executor`, `scene_executor`, `security_executor`, `timer_executor`, and `automation_executor`
- **Domain validation**: Each executor now declares `_ALLOWED_DOMAINS` and validates resolved entities belong to the correct domain before proceeding; wrong-domain entities are treated as "not found"
- **Safety net**: Prevents cross-domain entity leakage (e.g., `media_player.wohnzimmer_tv` being returned for a climate query about "Wohnzimmer")

New tests: test_action_executor (7) = 7 new tests
Updated tests: test_action_executor (2) = 2 updated tests

### 0.13.1 -- Orchestrator Flow Bug Fixes

4 end-to-end defects fixed in the orchestrator streaming pipeline, multi-agent dispatch, conversation persistence, and HA websocket integration:

- **Streaming error propagation**: StreamToken now carries an error field; orchestrator sets has_error from actual agent errors instead of hard-coding False; route adapters (SSE, WS, dashboard) surface errors to clients; HA integration detects error tokens and falls back to REST
- **Multi-agent partial failure**: Failed parallel agent branches are tracked with agent ID and error message; has_error reflects reality; merged response includes partial failure note; response dict contains partial_failure metadata
- **Conversation persistence**: _store_turn() now persists to DB via ConversationRepository.insert() so admin conversation pages and analytics totals reflect runtime conversations; DB failure is non-fatal (logged, does not break runtime)
- **HA WebSocket close/error handling**: _process_via_ws() raises on mid-stream CLOSED/ERROR instead of returning partial speech as success; triggers existing REST fallback path

New tests: test_agents (5), test_api (2), test_ha_client (4) = 11 new tests
Updated tests: test_agents (2) = 2 updated tests

### 0.13.0 -- Lower-Risk Hardening

4 lower-confidence risks and maintainability items addressed:

- **HA WebSocket concurrency**: Added `asyncio.Lock` to serialize overlapping conversation turns on the same entity, preventing response interleaving on the shared WebSocket connection
- **MCP transport/timeout**: Per-server timeout now wired to tool calls and connection establishment; removed misleading `http` transport alias (use `sse` instead); README updated to document actual transport support (stdio + SSE)
- **SQLite read/write split**: Separated `get_db()` into `get_db_read()` (no lock, concurrent reads) and `get_db_write()` (locked, serialized writes); all 86 repository call sites classified and updated; leverages existing WAL mode for true concurrent read access
- **Plugin trust boundary**: Removed deprecated `PluginContext.app` escape hatch (now raises `AttributeError`); added `event_bus` attribute to `PluginContext`; documented trust model in plugin-development.md

New tests: test_ha_client (1), test_mcp (5), test_db (2), test_plugins (3) = 11 new tests

### 0.12.1 -- Review Bug Fixes

5 confirmed defects fixed from full project review:

- **HA WebSocket startup**: Fixed `run()` never entering its main loop from a fresh client; `_running` is now set to `True` at the start of `run()` so the initial connection attempt proceeds
- **Plugin discovery isolation**: Disabled plugins no longer execute module-level code or constructor side effects during discovery; the DB-backed enabled check now runs before import
- **Settings type preservation**: Bulk settings route no longer silently rewrites `value_type` to `"string"`; `ON CONFLICT` clause now preserves existing metadata
- **Single-setting route hardening**: Single-setting PUT route now enforces the same allowlist and type validation as the bulk route; rejects unknown keys and invalid values
- **WebSocket auth deprecation**: Query-string API key auth is now deprecated with a logged warning; `Authorization` header is checked first; documentation updated

New tests: test_ha_client (3), test_plugins (3), test_db (2), test_api (4), test_security (6) = 18 new tests

### 0.12.0 -- Send-Agent & Sequential Orchestrator Dispatch

New send-agent enables content delivery to smartphones (via HA notify) and satellites (via TTS):

- **Send Agent**: New domain agent (`send.py`) delivers LLM-researched content to target devices
- **Sequential Dispatch**: Orchestrator supports 2-step sequential flow: content agent researches, send-agent delivers. New `[SEQ]` classification marker and `_handle_sequential_send()` method
- **Device Mapping**: New `send_device_mappings` DB table (migration 12) maps user-friendly names ("Laura Handy") to HA service targets (`notify.mobile_app_*` or `media_player.*`)
- **Dual Delivery Channels**: notify.* for smartphone push notifications, tts.speak for satellite speakers
- **Dashboard Page**: New "Send Devices" page with CRUD for device mappings and HA entity discovery
- **HA Service Discovery**: New `get_services()` on HA client; dashboard auto-discovers available notify and media_player targets
- **Content Formatting**: Send-agent uses LLM to format content per channel (full text for notify, spoken summary for TTS)
- **Target Name Parsing**: Regex-based extraction supports German ("sende an X") and English ("send to X")
- **Filler Language Fix**: Filler now uses full language names ("German (Deutsch)" instead of "de") and neutral user message to ensure correct language output
- **Filler Span Timestamps**: Filler spans now record actual generation timestamps for correct Span Timeline ordering
- **UI Cosmetics**: Navbar icon refresh (Agents/Custom Agents/Presence), app renamed to HA-AgentHub, flush button confirmations, analytics card height fix
- 22 new tests (847 total)

### 0.11.0 -- Prompt & LLM Architecture Restructuring

12 improvements to prompt consistency, code quality, and LLM call architecture:

- **PHASE Headers**: All 14 prompt files now have consistent header blocks indicating their role in the pipeline
- **Filler Message Role**: filler.txt split from single user message to proper system+user message pair, consistent with all other LLM calls
- **Prompt Deduplication**: New personality_base.txt shared include for mediate.txt, merge.txt, and rewrite.txt; eliminates triplicated personality/entity-preservation rules
- **Language Directives**: Domain agents (actionable + general) now inject explicit language directives for non-English users at runtime
- **Language Propagation**: TaskContext.language now properly propagated from orchestrator to domain agents
- **Classification History**: Conversation history in classification is now proper multi-turn messages instead of bracketed single-message bundle
- **Mediation Config Clarity**: Scattered inline overrides for mediation/merge (model, temperature, max_tokens) consolidated into _load_mediation_config() with pre-loaded attributes; default max_tokens increased to 8192 for reasoning model compatibility
- **Error Handling**: Agent errors now skip mediation and response caching; error metadata propagated in response dict
- **Cache Post-Mediation**: Response cache now stores post-mediation speech instead of raw agent response; rewrite agent focuses purely on phrasing variation
- **Original Text in Agent Calls**: Domain agents receive both condensed task and original user text when they differ
- **Orchestrator Code Dedup**: Extracted _do_cache_lookup(), _handle_response_cache_hit(), _store_response_cache(), _create_trace() helpers used by both handle_task() and handle_task_stream()
- **Span Boilerplate Reduction**: _NoOpSpan class and _optional_span() context manager eliminate if/else span_collector branches throughout orchestrator
- **Multi-Agent Streaming**: Progressive status markers (status="multi_agent" with agent list) yielded before non-streaming multi-agent fallback
- **Brevity Rules**: mediate, merge, and rewrite prompts now instruct "aim for 2-3 sentences"
- **Mediation Output Clarity**: mediate and merge prompts explicitly prohibit echoing "User asked:" preamble
- **Reasoning Effort**: New per-agent reasoning_effort setting (Low/Medium/High) in agent config dashboard; passed to litellm with drop_params=True for graceful fallback on unsupported models

### 0.10.0 -- Orchestrator Filler/Interim Responses

- Filler/interim TTS responses for slow agents: when an agent (e.g., general-agent with web search) takes longer than a configurable threshold, a short LLM-generated filler sentence is streamed to the client for immediate TTS playback
- New `expected_latency` field on AgentCard model (low/medium/high) to identify slow agents; general-agent set to "high"
- New `is_filler` field on StreamToken model to distinguish filler tokens from real content
- New `language` field on TaskContext model for language propagation through the pipeline
- Filler prompt template (prompts/filler.txt) generates personality-aware, language-matched filler sentences
- Configurable via DB settings: filler.enabled (default: false), filler.threshold_ms (default: 2000), filler.model (default: rewrite model)
- Race pattern in orchestrator handle_task_stream: races agent dispatch against threshold timer using async generator __anext__ with timeout
- HA custom component handles filler tokens via tts.speak service for immediate TTS playback, separate from main response
- Analytics/tracing span for filler events
- SSE and WebSocket endpoints pass through is_filler field
- Feature is disabled by default
- 15 new tests for filler logic, model fields, and API passthrough

### 0.9.4 -- Conversation Memory for Follow-up Questions

- Inject conversation history into orchestrator classification prompt so follow-up questions are routed with context
- Annotate stored conversation turns with agent_id for agent-level context tracking
- Generate fallback conversation_id when Home Assistant sends None
- Updated orchestrator prompt with conversation context routing rules
- 6 new tests for conversation memory, history injection, and agent_id annotation

### 0.9.3 -- Agent Description & Skills Improvements

- Improved all 9 agent descriptions for better orchestrator LLM routing accuracy
- light-agent: added switch_control, toggle, illuminance_sensor skills
- music-agent: added shuffle, repeat skills
- timer-agent: added timer_pause, timer_resume, alarm skills
- climate-agent: added climate_on_off, weather_sensor skills
- scene-agent: fixed misleading "manages" wording
- security-agent: added door_sensor, window_sensor, doorbell, smoke_sensor, camera_control skills
- media-agent: added mute skill, explicit distinction from music-agent
- general-agent: added web_search, current_events, conversation skills
- No behavioral or code logic changes -- description/skills text only

### 0.9.2 -- MCP UI Improvements

- Hide Delete button for built-in MCP servers (DuckDuckGo) with API-level protection (403)
- "Built-in" badge shown next to built-in server names on MCP Servers page
- MCP tool assignment UI in agent edit form with checkbox-based tool selection
- MCP server badge on agent cards showing assigned tools per server (purple badges)
- New API endpoint: GET /api/admin/mcp-servers/agent-tools-summary for bulk assignment data

### 0.9.1 -- Docker Image Size Optimization

- Switched to CPU-only PyTorch in Docker build, eliminating ~4.3 GB of unused NVIDIA/CUDA/Triton libraries
- Docker image reduced from ~9.31 GB to ~5.0 GB (46% reduction)
- No functional changes -- sentence-transformers and all other packages continue to work identically

### 0.9.0 -- Web Search via MCP + LLM Tool Calling

- DuckDuckGo web search MCP server (bundled, stdio transport, zero API key)
  - `web_search` tool: general web search with configurable max results
  - `web_search_news` tool: news-specific search with date and source info
- LLM tool/function calling support via new `complete_with_tools()` in llm/client.py
  - Configurable max tool rounds to prevent infinite loops
  - Automatic tool_choice="auto" -- LLM decides when to use tools
- MCP tool assignment for built-in agents (new `agent_mcp_tools` DB table)
  - Admin API endpoints for assigning/unassigning MCP tools to any agent
- GeneralAgent web search integration
  - Fetches assigned MCP tools and uses tool calling when tools are available
  - Falls back to plain LLM completion when no tools assigned
- Auto-registration: DuckDuckGo MCP server registered on first startup, tools auto-assigned to general-agent
- Updated general agent prompt with web search usage guidelines
- New files: mcp/servers/duckduckgo_server.py, mcp/servers/__init__.py
- 9 new tests for tool calling, MCP server, agent integration, and tool assignment

### 0.8.0 -- Domain Agent Status/State Query Capabilities

- Read-only status query support for all 7 domain agents (light, climate, automation, scene, security, music, media)
- Each agent now supports querying individual entity status and listing all entities in its domain
- Light agent: query_light_state and list_lights actions (covers both light.* and switch.* entities)
- Climate agent: query_climate_state and list_climate actions (includes climate sensors)
- Automation agent: query_automation_state and list_automations actions (shows enabled/disabled status and last triggered)
- Scene agent: query_scene and list_scenes actions
- Security agent: query_security_state and list_security actions (locks, alarms, cameras, binary sensors with device_class awareness)
- Music agent: query_music_state and list_music_players actions (track, artist, volume info)
- Media agent: query_media_state and list_media_players actions (source, volume, playback info)
- Updated all agent card descriptions and skills for improved orchestrator routing of query requests
- Updated all domain prompts to instruct LLM to use JSON action blocks for status queries
- 42 new tests for query/list actions across all 7 domains

### 0.7.0 -- Timer Notification System & Alarms Dashboard

- Timer & Alarms Dashboard: new /dashboard/timers page with active timers, alarms, timer pool, delayed tasks overview
- Device context propagation: device_id/area_id from HA voice satellite through entire processing pipeline
- TimerMetadata tracking in timer pool (origin device, area, media_player association)
- WebSocket timer.finished/timer.cancelled event listener for real-time timer completion detection
- Multi-channel notification dispatcher: TTS on origin satellite, persistent_notification, mobile push
- LLM-generated interactive TTS messages with conversation continuation on the originating satellite
- AlarmMonitor background task for input_datetime entities (30s polling, daily deduplication)
- Recently expired timer tracking with snooze-last-expired fallback
- Notification profile settings configurable via admin API
- New files: notification_dispatcher.py, alarm_monitor.py, timers.html dashboard template
- New API endpoints: GET /dashboard/timers, GET /api/admin/timers, GET/PUT /api/admin/notification-profile, GET /api/admin/alarm-monitor, GET /api/admin/timers/recently-expired
- 11 modified files across container/app/ and custom_components/

### 0.6.0 -- New Agents & Container Hardening

- New agents: timer-agent (timer start/stop/pause + alarms), scene-agent (scene activation), automation-agent (enable/disable/trigger), media-agent (TV, Chromecast, generic media_player control)
- Converted all 4 agents from BaseAgent stubs to full ActionableAgent implementations with domain-specific executors
- Container hardening: multi-stage Dockerfile, non-root user (gosu entrypoint), pinned dependencies, resource limits (2 CPU / 2GB RAM), health check
- Security: prompt injection defenses in all actionable agents, trace metadata sanitization, plugin sandboxing, WebSocket/REST input size limits
- HA integration fix: removed deprecated async_process, modern ConversationEntity pattern, assist_pipeline dependency
- HA options flow for reconfiguration without re-adding integration
- Background WebSocket reconnection with exponential backoff
- A2A dispatch timeout (120s), cache browse pagination, server-side WebSocket heartbeat
- Fernet key backup mechanism, graceful shutdown improvements
- HTTPS deployment docs, backup/restore documentation
- 26 new tests (716 total)

- Full code review with 21 fixes across 4 phases (critical, security, performance, architecture)
- Critical: Fixed unbounded conversation memory leak in orchestrator (TTL eviction + max count)
- Security: XSS prevention in setup wizard, session cookie secure flag, settings allowlist validation
- Security: WebSocket API key moved from query param to Authorization header
- Performance: Async ChromaDB wrappers via asyncio.to_thread() to unblock event loop
- Performance: Shared SQLite connection pool replacing per-operation connect/close
- Performance: Interval-based LRU cache eviction with batch deletes, buffered hit count updates
- Architecture: ActionableAgent base class extracting duplicated code from 4 domain agents
- Fixes: TracingMiddleware UnboundLocalError guard, Fernet decryption error handling, cache model class ordering, duplicate admin account upsert
- Centralized version string in container/app/__init__.py
- 16 new tests for setup wizard, security auth, conversation memory eviction, cache eviction

### 0.4.0 -- Phase 4: Testing, CI/CD & Documentation

- Comprehensive test suite with pytest + pytest-asyncio (300+ tests covering config, models, DB, A2A, entity matching, cache, agents, HA client, LLM, MCP, plugins, presence, middleware, security, and integration tests)
- CI/CD pipelines: GitHub Actions for test, lint, Docker build, HACS validation, and release
- README rewrite with features, architecture overview, quick start guide
- Documentation: deployment guide, configuration reference, architecture overview, API reference, troubleshooting guide
- Ruff linter and formatter configuration

### 0.3.0 -- Phase 3: Dashboard, Analytics & Plugin System

- Admin dashboard with 14 pages (overview, agents, custom agents, MCP servers, entity visibility, entity index, presence, rewrite, conversations, cache, analytics, traces, system health, plugins)
- Analytics collection and Chart.js charting (request counts, cache hit rates, latency, token usage)
- Trace span collection and vis.js Timeline Gantt visualization
- Plugin system with lifecycle hooks and event bus
- HACS-ready packaging

### 0.1.0 -- Initial Scaffolding

- Project scaffolding and directory structure
- Project definition document

## Recent Changes (since 0.21.1)

### 0.21.2 -- Orchestrator condensed-task hardening

- `_parse_classification` strips embedded `<known-agent> (NN%):`
  fragments and collapses verbatim repetitions, so malformed LLM
  output no longer poisons the task forwarded to specialized agents
  or the routing cache.
- `_classify` injects a per-request `{language_hint}` into the
  orchestrator prompt; for non-English requests the LLM is explicitly
  told to copy entity/room names verbatim. Few-shot examples remain
  English-only.
- `RoutingCache.lookup` rejects pre-existing entries whose
  `condensed_task` still contains an embedded classification
  fragment, so legacy corruption self-heals on next access.
