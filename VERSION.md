# Version

**Current Version:** 0.12.0

## Version History

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
