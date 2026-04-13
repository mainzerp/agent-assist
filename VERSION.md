# Version

**Current Version:** 0.6.0

## Version History

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

## Recent Changes (since 0.6.0)

### Timer Agent Extensions
- 10 new actions: query_timer, list_timers, snooze_timer, list_alarms, start_timer_with_notification, delayed_action, sleep_timer, create_reminder, create_recurring_reminder, named timer pool
- DelayedTaskManager for post-timer callbacks (notifications, delayed actions, sleep timer)
- Timer pool management with auto-allocation of idle timer helpers
- Calendar integration for reminders with rrule support
- 11 new tests (727 total)
