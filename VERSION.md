# Version

**Current Version:** 0.18.20

## Version History

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

## Recent Changes (since 0.18.4)

(none yet)
