# Version

**Current Version:** 0.14.4

## Version History

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

## Recent Changes (since 0.8.0)

(none yet)
