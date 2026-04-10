# agent-assist -- Implementation TODO

A complete implementation checklist for the agent-assist multi-agent AI assistant for Home Assistant. Organized into four development phases, ordered by dependency within each group. Every deliverable from the project definition is covered.

---

## Phase 0: Project Setup & Infrastructure

Cross-cutting scaffolding, configuration, dependencies, Docker, authentication, models, prompts, and foundational architecture that all later phases depend on.

### 0.1 Project Scaffolding

- [x] Create top-level directory structure matching Project Structure definition (`container/app/`, `custom_components/agent_assist/`, `docs/`, `data/`, `plugins/`, `tests/`)
- [x] Create `README.md` with project overview, setup instructions, and architecture summary
- [x] Create `LICENSE` file (choose and apply license)
- [x] Create `VERSION.md` with initial version `0.1.0` and changelog skeleton

### 0.2 SQLite Schema & Configuration

- [x] Create SQLite database initialization module in `container/app/db/schema.py` (aiosqlite connection management, startup schema creation, table definitions)
- [x] Define `settings` table schema (key TEXT PRIMARY KEY, value TEXT, category TEXT) -- replaces config.yaml
- [x] Define `agent_configs` table schema (agent_id TEXT PRIMARY KEY, enabled BOOLEAN, model TEXT, timeout INTEGER, max_iterations INTEGER, temperature REAL, max_tokens INTEGER) -- replaces agents.yaml
- [x] Define `custom_agents` table schema (name TEXT PRIMARY KEY, description TEXT, system_prompt TEXT, model_override TEXT, mcp_tools TEXT, entity_visibility TEXT, intent_patterns TEXT) -- replaces custom_agents.yaml
- [x] Define `aliases` table schema (alias TEXT PRIMARY KEY, entity_id TEXT) -- replaces entity_matching.yaml aliases section
- [x] Define `entity_matching_config` table schema (key TEXT PRIMARY KEY, value TEXT) -- replaces entity_matching.yaml weights/thresholds
- [x] Define `mcp_servers` table schema (name TEXT PRIMARY KEY, transport TEXT, command_or_url TEXT, env_vars TEXT, timeout INTEGER) -- replaces mcp_servers.yaml
- [x] Define `secrets` table schema (key TEXT PRIMARY KEY, encrypted_value BLOB) -- Fernet-encrypted secrets storage
- [x] Define `admin_accounts` table schema (username TEXT PRIMARY KEY, password_hash TEXT)
- [x] Define `setup_state` table schema (key TEXT PRIMARY KEY, value TEXT) -- tracks setup wizard completion
- [x] Create `container/app/db/repository.py` with async CRUD operations via aiosqlite
- [x] Create `container/app/db/migrations.py` for schema versioning
- [x] Implement seed data insertion (default settings, default agent configs, default entity matching weights)
- [x] Create `.env.example` documenting infrastructure-only environment variables with defaults (`CONTAINER_HOST=0.0.0.0`, `CONTAINER_PORT=8080`, `LOG_LEVEL=INFO`, `CHROMADB_PERSIST_DIR=/data/chromadb`, `SQLITE_DB_PATH=/data/agent_assist.db`) with comments noting that no secrets (API keys, tokens, passwords) belong in env vars
- [x] Implement Pydantic Settings configuration loader in `container/app/config.py` (load infrastructure env vars first -- `CONTAINER_HOST`, `CONTAINER_PORT`, `LOG_LEVEL`, `CHROMADB_PERSIST_DIR`, `SQLITE_DB_PATH` -- then SQLite for all persistent configuration; no fail-fast on HA/LLM vars since those come from the setup wizard; expose typed config)

### 0.3 Docker & Deployment

- [x] Write `Dockerfile` for the container (Python base image, copy app, install deps, expose port, CMD uvicorn)
- [x] Write `docker-compose.yml` with agent-assist service, volume mounts for ChromaDB persistence and SQLite data, network config for HA access, and infrastructure-only environment variables (`CONTAINER_HOST`, `CONTAINER_PORT`, `LOG_LEVEL`, `CHROMADB_PERSIST_DIR`, `SQLITE_DB_PATH`) -- no secrets in compose file
- [x] Configure Docker volume for ChromaDB persistence directory (`/data/chromadb`)
- [x] Add Docker health check using `/api/health` endpoint with restart policy
- [x] Configure Uvicorn as ASGI server in Docker CMD (host, port, workers)

### 0.4 Authentication & Security

- [x] Implement shared secret / API key authentication for HA integration to container communication
- [x] Implement Long-Lived Access Token authentication for container to HA REST API calls
- [x] Implement Fernet-encrypted token storage in SQLite (HA token, LLM API keys)
- [x] Implement admin password hashing (bcrypt) for dashboard authentication
- [x] Implement auth middleware for dashboard and admin routes (password-based login)
- [x] Implement Fernet encryption utility for secrets storage (encrypt/decrypt API keys, tokens, passwords in SQLite)
- [x] Ensure no tokens are logged or exposed in error messages
- [x] Implement input sanitization and prompt injection prevention for all incoming requests
- [x] Implement container API authentication validation middleware
- [x] Implement auto-reconnect with exponential backoff (max 60s) on WebSocket disconnect

### 0.5 Pydantic Models

- [x] Create conversation request/response models (including streaming types) in `container/app/models/conversation.py`
- [x] Create agent configuration models in `container/app/models/agent.py`
- [x] Create cache entry models (including cached action data, metadata) in `container/app/models/cache.py`
- [x] Create entity index entry models in `container/app/models/entity_index.py`

### 0.6 LLM Integration

- [x] Integrate litellm as unified LLM interface in `container/app/` (import, configure, test)
- [x] Configure OpenRouter provider support (single API key, model routing)
- [x] Configure Groq provider support (fast inference for orchestrator and rewrite agent)
- [x] Configure Ollama provider support (local models, no API key)
- [x] Implement per-agent model configuration (model name, max tokens, temperature) loaded from SQLite `agent_configs` table

### 0.7 Async Architecture

- [x] Ensure entire stack is async: FastAPI endpoints, aiohttp/httpx HTTP clients, asyncio agent execution, WebSocket connections -- no blocking I/O on main thread

### 0.8 Dependencies

- [x] Create `container/requirements.txt` with all core dependencies:
  - [x] FastAPI + Uvicorn
  - [x] litellm
  - [x] sentence-transformers (all-MiniLM-L6-v2)
  - [x] ChromaDB (embedded mode)
  - [x] rapidfuzz + pyphonetics
  - [x] aiohttp / httpx
  - [x] Pydantic v2
  - [x] MCP SDK (Python)
  - [x] Jinja2 (for HTMX templates)
  - [x] SQLite + aiosqlite (required, primary persistent store for all structured data)

### 0.9 Agent Prompts

- [x] Create `container/app/prompts/orchestrator.txt` -- routing instructions, agent descriptions, name preservation rule
- [x] Create `container/app/prompts/music.txt` -- domain-scoped, under 200 tokens
- [x] Create `container/app/prompts/light.txt` -- domain-scoped, under 200 tokens
- [x] Create `container/app/prompts/rewrite.txt` -- phrasing variation instructions
- [x] Create `container/app/prompts/general.txt` -- fallback/Q&A, under 200 tokens
- [x] Enforce name preservation rule in ALL prompts: never translate, paraphrase, or normalize entity/device/room/location names
- [x] Ensure all agent prompts are under 200 tokens and scoped to their domain

### 0.10 Testing Scaffold

- [x] Create `container/tests/` directory structure with placeholder test files for each major module

### 0.11 Graceful Degradation

- [x] Implement fallback: if embedding cache unavailable, bypass cache and use LLM directly
- [x] Implement fallback: if rewrite agent fails, return cached response verbatim
- [x] Implement fallback: if entity index unavailable, fall back to hybrid matcher without embedding signal
- [x] Implement fallback: REST fallback when WebSocket is unavailable

### 0.12 Network & Communication Configuration

- [x] Configure container host/port settings (loaded from config)
- [x] Configure HA URL setting (loaded from config)
- [x] Configure streaming mode selection: `websocket` (default), `sse`, or `none`
- [x] Configure WebSocket reconnect interval setting
- [x] Configure stream buffer size (token batching) setting

### 0.13 Dashboard Skeleton

- [x] Create Jinja2 templates directory at `container/app/dashboard/templates/`
- [x] Create static files directory at `container/app/dashboard/static/` (CSS, JS, HTMX)
- [x] Create dashboard routes file at `container/app/dashboard/routes.py`

---

## Phase 1: MVP

Working end-to-end conversation flow from HA to container and back, with A2A protocol as the core agent communication mechanism and agents executing HA actions directly.

### 1.1 HA Custom Integration (I/O Bridge)

- [x] Create integration file structure: `custom_components/agent_assist/__init__.py`, `manifest.json`, `config_flow.py`, `const.py`, `conversation.py`, `strings.json`, `translations/en.json`
- [x] Implement `ConversationEntity` subclass in `conversation.py` that registers as a conversation agent in HA
- [x] Implement `_async_handle_message()` to forward user text input and conversation metadata to the Docker container
- [x] Register `ConversationEntityFeature.CONTROL` to indicate device control capability
- [x] Implement config flow with UI-based configuration (container URL, auth token)
- [x] Implement WebSocket streaming communication to Docker container (persistent connection)
- [x] Implement token-level streaming: receive response tokens from container and deliver to user via TTS/text
- [x] Assemble streamed tokens into `ConversationResult` for HA

### 1.2 FastAPI Container

- [x] Create FastAPI application entry point in `container/app/main.py` (app factory, startup/shutdown events, middleware)
- [x] Implement WebSocket streaming endpoint `/ws/conversation` in `container/app/api/routes/conversation.py`
- [x] Implement SSE streaming fallback endpoint `/api/conversation/stream` in `container/app/api/routes/conversation.py`
- [x] Implement REST fallback endpoint `/api/conversation` for non-streaming full-response mode
- [x] Implement admin API endpoints `/api/admin/*` in `container/app/api/routes/admin.py`
- [x] Implement health check endpoint `/api/health` in `container/app/api/routes/health.py`
- [x] Implement basic structured logging (request IDs, timestamps, log levels)

### 1.3 A2A Protocol Layer (JSON-RPC 2.0)

- [x] Define JSON-RPC 2.0 message types and A2A envelope in `container/app/a2a/protocol.py`
  - [x] Request model: `jsonrpc`, `method`, `params`, `id` fields
  - [x] Response model: `jsonrpc`, `result`, `error`, `id` fields
  - [x] Error model: standard JSON-RPC 2.0 error codes including `-32000` timeout
- [x] Define agent card schema as Pydantic model (agent_id, name, description, skills, input/output types, endpoint)
- [x] Implement agent registry in `container/app/a2a/registry.py`
  - [x] `register(agent_card)` -- register an agent card
  - [x] `discover(agent_id)` -- return agent card for given ID
  - [x] `list_agents()` -- return all registered agent cards
- [x] Implement `message/send` method -- synchronous request-response dispatch to target agent
- [x] Implement `message/stream` method -- streaming response dispatch with async generator
- [x] Implement `agent/discover` and `agent/list` protocol methods
- [x] Implement in-process async transport in `container/app/a2a/transport.py` (direct async function calls wrapped in A2A format, near-zero overhead)
- [x] Implement transport abstraction layer with interface for future HTTP transport
- [x] Implement message dispatcher in `container/app/a2a/dispatcher.py` (routes incoming A2A messages to the correct agent via registry lookup)
- [x] Implement standard JSON-RPC 2.0 error handling (parse error, invalid request, method not found, timeout `-32000`)

### 1.4 Orchestrator Agent

- [x] Implement orchestrator agent in `container/app/agents/orchestrator.py`
- [x] Implement intent classification using Groq for fast routing
- [x] Implement task condensation -- create condensed task for target agent preserving all entity/device/room/location names verbatim
- [x] Implement A2A dispatch via `message/send` to selected specialized agent
- [x] Implement streaming response relay back to HA Integration via WebSocket
- [x] Implement conversation context management (last 2-3 turns)
- [x] Create orchestrator system prompt in `prompts/orchestrator.txt` (routing instructions, agent descriptions, name preservation mandate)
- [x] Implement context propagation: condensed task, user_text (unmodified), domain-specific entity states, conversation turns, presence data (when available)

### 1.5 Specialized Agents (Phase 1 Set)

- [x] Implement base agent class in `container/app/agents/base.py`
  - [x] Inject HA REST API client reference
  - [x] Inject entity index reference for vector search
  - [x] Define abstract `handle_task(task)` method
  - [x] Implement A2A agent card generation from class metadata
- [x] Implement `light-agent` in `container/app/agents/light.py`
  - [x] Handle on/off, brightness, color, color temperature commands
  - [x] Resolve target entity via entity index vector search
  - [x] Call HA REST API `POST /api/services/light/turn_on`, `turn_off` with appropriate service_data
  - [x] Verify action result by checking entity state after execution
  - [x] Return natural language response with action outcome
- [x] Implement `general-agent` in `container/app/agents/general.py`
  - [x] Handle unroutable requests and general Q&A
  - [x] Use LLM for freeform responses (no HA service calls)
- [x] Implement `music-agent` in `container/app/agents/music.py`
  - [x] Handle play/pause/skip, playlist selection, volume commands
  - [x] Resolve target media_player entity via entity index
  - [x] Call HA REST API media_player services directly
  - [x] Return natural language response with action outcome
- [x] Register all 3 agents with A2A agent cards at container startup
- [x] Write system prompt files: `prompts/light.txt`, `prompts/music.txt`, `prompts/general.txt` (each under 200 tokens, domain-scoped)
- [x] Configure LLM calls via OpenRouter for all specialized agents using litellm
- [x] Enforce name preservation: agents must use entity/device/room names exactly as stated in the condensed task

### 1.6 Pre-Embedded Entity Index

- [x] Implement ChromaDB collection for entity index (separate from cache collections) in `container/app/entity/index.py`
- [x] Implement startup population -- fetch all HA entities via REST API and embed into index (friendly_name, area, domain, device_class)
- [x] Implement vector search interface for entity resolution by agents
- [x] Configure entity attributes to embed: friendly_name, area, domain, device_class (from SQLite `settings` table)

### 1.7 Two-Tier Embedding Cache

- [x] Implement routing cache -- ChromaDB collection mapping user text embeddings to agent routing decisions in `container/app/cache/routing_cache.py`
- [x] Implement response cache -- ChromaDB collection mapping user text + context to responses + cached HA actions in `container/app/cache/response_cache.py`
- [x] Implement local sentence-transformers embedding (`all-MiniLM-L6-v2`, 384-dim) in `container/app/cache/embedding.py`
- [x] Implement external embedding provider support via litellm in `container/app/cache/embedding.py`
- [x] Implement unified embedding interface (local or external, configurable) in `container/app/cache/embedding.py`
- [x] Implement routing cache threshold logic: > 0.92 hit (skip intent classification), < 0.92 miss (run full classification)
- [x] Implement response cache threshold logic: > 0.95 hit, 0.80-0.95 partial match, < 0.80 miss
- [x] Implement response cache hit direct execution -- execute cached HA action via REST API on hit, stream response
- [x] Implement reactive invalidation -- on cached action execution failure, remove failed cache entry and re-route through full pipeline
- [x] Implement LRU eviction (configurable max entries per tier: 50,000 routing, 20,000 response)
- [x] Implement persistent storage to Docker volume (no TTL, entries persist across restarts)
- [x] Implement cache entry metadata: timestamp, last-accessed, hit count, entity IDs, agent, cached action data
- [x] Implement ChromaDB wrapper managing cache tiers + entity index collections in `container/app/cache/vector_store.py`
- [x] Implement cache manager with invalidation logic, unified interface in `container/app/cache/cache_manager.py`

### 1.8 Basic Entity Matching

- [x] Implement Levenshtein distance matching in `container/app/entity/matcher.py`
- [x] Implement alias resolution (SQLite-based user-defined mappings from `aliases` table) in `container/app/entity/aliases.py`
- [x] Implement entity index vector search signal (query pre-embedded entity index for embedding similarity)

### 1.9 HA REST API Client

- [x] Implement async REST API client in `container/app/ha_client/rest.py` (GET /api/states, POST /api/services/<domain>/<service>, POST /api/events/<event_type>)
- [x] Implement Long-Lived Access Token authentication in `container/app/ha_client/auth.py`
- [x] Wire HA client into base agent class for direct action execution by specialized agents
- [x] Wire HA client into container startup for entity index population

### 1.10 Setup Wizard

- [x] Implement setup-incomplete state detection (check SQLite `setup_state` table for completion flag)
- [x] Create setup-required redirect middleware (redirect all non-setup routes to /setup/ when unconfigured)
- [x] Create setup wizard base template (HTMX + Jinja2, step indicator, progress bar)
- [x] Implement Step 1: Admin password creation (bcrypt hashing, store encrypted)
- [x] Implement Step 2: HA connection configuration (URL + token input, live connection test endpoint)
- [x] Implement Step 3: Container API key generation (auto-generate, display once, store encrypted)
- [x] Implement Step 4: LLM provider key entry (OpenRouter, Groq, Ollama URL fields, live test endpoint)
- [x] Implement Step 5: Review & complete (summary page, confirm button, trigger initialization)
- [x] Create HA connection test endpoint (POST /setup/test/ha -- validates URL and token)
- [x] Create LLM provider test endpoint (POST /setup/test/llm -- validates key with a small request)
- [x] Implement post-wizard initialization (trigger entity index population, cache setup, agent registration)
- [x] Implement Fernet encryption utility for secrets (encrypt/decrypt API keys, tokens, passwords)

---

## Phase 2: Extended Agents, Entity Matching, Rewrite Agent & MCP

Full agent coverage, robust caching with response rewriting, hybrid entity matching, and MCP tool support.

### 2.1 Additional Specialized Agents

- [x] Implement `timer-agent` in `container/app/agents/timer.py`
  - [x] Handle set/cancel timers, reminders
  - [x] Create system prompt `prompts/timer.txt` (under 200 tokens)
- [x] Implement `climate-agent` in `container/app/agents/climate.py`
  - [x] Handle temperature, mode, fan speed, humidity commands
  - [x] Resolve target entity via entity index, call HA climate services directly
  - [x] Create system prompt `prompts/climate.txt` (under 200 tokens)
- [x] Implement `media-agent` in `container/app/agents/media.py`
  - [x] Handle TV, speakers, casting, playback control commands
  - [x] Resolve target entity via entity index, call HA media_player services directly
  - [x] Create system prompt `prompts/media.txt` (under 200 tokens)
- [x] Implement `scene-agent` in `container/app/agents/scene.py`
  - [x] Handle activate/deactivate scenes
  - [x] Call HA scene services directly
  - [x] Create system prompt `prompts/scene.txt` (under 200 tokens)
- [x] Implement `automation-agent` in `container/app/agents/automation.py`
  - [x] Handle enable/disable automations, trigger
  - [x] Call HA automation services directly
  - [x] Create system prompt `prompts/automation.txt` (under 200 tokens)
- [x] Implement `security-agent` in `container/app/agents/security.py`
  - [x] Handle locks, cameras, alarm system commands
  - [x] Call HA lock/alarm services directly
  - [x] Create system prompt `prompts/security.txt` (under 200 tokens)
- [x] Register all additional agents as A2A agents with agent cards at startup

### 2.2 Rewrite Agent

- [x] Implement rewrite agent in `container/app/agents/rewrite.py`
  - [x] Accept cached response text, produce rephrased version preserving meaning
  - [x] Use small/fast LLM call (Groq) or template-based variation
- [x] Make rewrite agent configurable enable/disable via SQLite `settings` table
- [x] Enforce target latency < 100ms for rewrite operations
- [x] Create rewrite agent prompt `prompts/rewrite.txt` (phrasing variation only, no meaning changes)
- [x] Integrate rewrite agent into response cache hit path (optional step after direct execution)

### 2.3 Full Hybrid Entity Matching Engine

- [x] Implement individual signal modules in `container/app/entity/signals.py`
  - [x] Levenshtein distance signal (using rapidfuzz)
  - [x] Jaro-Winkler similarity signal (using rapidfuzz)
  - [x] Phonetic matching signal using Soundex and Metaphone (using pyphonetics)
  - [x] Embedding similarity signal that queries the pre-embedded entity index
  - [x] Alias lookup signal from SQLite-based alias store
- [x] Implement weighted scoring system in `container/app/entity/matcher.py`
  - [x] Accept configurable per-signal weights (defaults: Levenshtein 0.2, Jaro-Winkler 0.2, Phonetic 0.15, Embedding 0.3, Alias 0.15)
  - [x] Compute weighted sum of all signal scores for each candidate entity
  - [x] Return top match if score exceeds confidence threshold (default: 0.75)
  - [x] Return top-N candidates (default: 3) for LLM disambiguation if no match exceeds threshold
- [x] Implement alias store in `container/app/entity/aliases.py`
  - [x] Load user-defined aliases from SQLite `aliases` table
  - [x] Support exact-match alias lookup (e.g., "nightstand lamp" -> `light.bedroom_nightstand`)
- [x] Seed SQLite `entity_matching_config` and `aliases` tables with default signal weights, confidence threshold, and example aliases

### 2.4 Entity Visibility Controls

- [x] Implement per-agent entity filtering (configurable entity domains/areas per agent)
- [x] Integrate HA exposed-entity list via REST API as pre-filtered baseline
- [x] Create admin dashboard page for entity visibility management (checkbox-based UI by domain and area)

### 2.5 MCP Tool Integration

- [x] Implement MCP client in `container/app/mcp/client.py` (connect to MCP servers, manage connections)
- [x] Implement stdio transport for local tool processes
- [x] Implement HTTP/SSE transport for remote MCP servers
- [x] Implement tool discovery after connecting to an MCP server
- [x] Implement tool-to-agent assignment (map discovered tools to specific agents)
- [x] Implement MCP server registry in `container/app/mcp/registry.py` (add/remove/list servers)
- [x] Implement MCP tools module in `container/app/mcp/tools.py` (tool invocation, result parsing)
- [x] Define MCP server storage in SQLite `mcp_servers` table with per-server config (name, transport, command/URL, env vars as JSON, timeout)

### 2.6 Runtime Agent Builder

- [x] Implement admin dashboard page for creating custom agents without code
- [x] Define custom agent schema: name, description, system prompt, model override, assigned MCP tools, entity visibility rules, supported intent patterns
- [x] Implement storage in SQLite `custom_agents` table with load-at-startup logic
- [x] Implement hot reload -- new/updated custom agents take effect without container restart
- [x] Implement A2A agent card registration for custom agents (dynamic registration in agent registry)
- [x] Implement orchestrator routing consideration for custom agents during intent classification
- [x] Provide HA client access for custom agents (same as built-in agents)
- [x] Implement dynamic loader for runtime-created agents in `container/app/agents/custom_loader.py`

### 2.7 Multi-Turn Conversation Support

- [x] Implement `conversation_id` tracking across agent switches (passed through A2A params)
- [x] Implement conversation memory/history with limited context window (last 2-3 turns)

### 2.8 Entity Index Real-Time Refresh

- [x] Implement HA WebSocket subscription for state change events in `container/app/ha_client/websocket.py`
- [x] Implement real-time entity index refresh (add new entities, prune removed entities, re-embed renamed entities)

### 2.9 Cache Improvements

- [x] Implement hot-reloadable cache thresholds via admin UI (changes take effect on next request without restart)

### 2.10 Agent Reliability

- [x] Implement agent timeout logic (3-5 seconds per agent call, configurable)
- [x] Implement fallback to general agent on timeout or agent failure
- [x] Implement max iterations per agent to prevent infinite loops (default: 3)

### 2.11 Improved Intent Routing

- [x] Implement embedding-based intent classification with routing cache (embedding similarity against known intent templates, LLM fallback for ambiguous requests)

### 2.12 Presence Detection

- [x] Implement sensor auto-discovery in `container/app/presence/sensors.py`
  - [x] Query HA REST API for motion_sensor, occupancy, and mmWave sensor entities
  - [x] Map discovered sensors to rooms/areas based on HA area assignments
- [x] Implement room-level confidence scoring in `container/app/presence/scoring.py`
  - [x] Weight by sensor type: mmWave (1.0) > occupancy (0.8) > motion (0.5)
  - [x] Factor in sensor recency (more recent triggers = higher confidence)
  - [x] Apply configurable presence decay timeout (default: 300 seconds)
- [x] Implement presence detection engine in `container/app/presence/detector.py`
  - [x] Aggregate sensor data to produce per-room presence confidence
  - [x] Expose current presence state for context enrichment
- [x] Configure sensor priority: mmWave > occupancy > motion
- [x] Inject presence data into orchestrator context propagation (detected room/area added to agent task context)
- [x] Implement configurable presence decay timeout in SQLite `settings` table

### 2.13 Observability

- [x] Implement structured logging with agent/request tracing (trace IDs, agent names, latencies per component)

---

## Phase 3: Dashboard, Analytics & Plugin System

Full admin UI, operational intelligence, plugin ecosystem, and production readiness.

### 3.1 Admin Dashboard UI

- [x] Evaluate and implement dashboard framework upgrade (SvelteKit or enhanced HTMX)
- [x] Implement agent configuration UI page (enable/disable agents, edit prompts, select models per agent)
- [x] Implement MCP server management UI page (add/remove servers, view discovered tools, assign tools to agents)
- [x] Implement Runtime Agent Builder UI page (full CRUD for custom agents: create, read, update, delete)
- [x] Implement entity visibility management UI page (checkbox-based, organized by domain and area)
- [x] Implement entity index status page (collection size, last refresh time, entity count per domain)
- [x] Implement presence detection status page (room occupancy map, sensor health indicators)
- [x] Implement rewrite agent configuration UI page (enable/disable, model selection, temperature slider)
- [x] Implement system health monitoring page (container status, HA connection, ChromaDB status, LLM provider health)
- [x] Implement traces page with detailt informations of every request and its involved agents and trace runtime as gantt diagramm
- [x] Implement overview dashboard with important metrics

### 3.2 Analytics

- [x] Implement request count tracking (total requests, per-agent breakdown)
- [x] Implement cache hit rate tracking per tier (routing cache hits/misses, response cache hits/partial/misses)
- [x] Implement direct execution rate metric (percentage of requests served entirely from cache)
- [x] Implement rewrite agent usage stats (invocation count, latency, skip rate)
- [x] Implement latency tracking per agent (average, p50, p95, p99)
- [x] Implement token usage tracking (tokens consumed per agent, per provider)
- [x] Implement analytics charts (SvelteKit charting library or HTMX-compatible charts for visual dashboards)


### 3.3 Conversation & Cache Management

- [x] Implement conversation log viewer (searchable, filterable by agent, date range, conversation_id)
- [x] Implement cache management UI (view entries, search by query text, flush per tier, inspect cached actions and metadata)

### 3.4 Plugin System

- [x] Implement plugin loader in `container/app/plugins/loader.py`
  - [x] Scan `plugins/` directory for Python scripts
  - [x] Load and validate plugin modules
  - [x] Manage plugin lifecycle (discover, load, configure, startup, ready, shutdown)
- [x] Define lifecycle hooks in `container/app/plugins/hooks.py`
  - [x] `configure()` -- called during config load phase
  - [x] `startup()` -- called during app startup phase
  - [x] `ready()` -- called when all agents are registered and system is ready
- [x] Implement base plugin class in `container/app/plugins/base.py`
  - [x] Abstract methods for lifecycle hooks
  - [x] Access to agent registry, MCP registry, event bus, dashboard routes
- [x] Implement plugin capabilities
  - [x] Register new A2A agents from plugin code
  - [x] Add MCP servers from plugin code
  - [x] Modify routing rules from plugin code
  - [x] Add dashboard pages from plugin code
  - [x] Register event handlers from plugin code
- [x] Create `plugins/README.md` with plugin development guide (how to create, lifecycle hooks, available APIs, example plugin)
- [x] Implement plugin discovery/management UI page in admin dashboard (enable/disable, view info, install from repository)
- [x] Write plugin development documentation in `docs/`

### 3.5 Production Readiness

- [x] Package HA custom integration as HACS-ready (ensure `manifest.json`, `hacs.json`, repository structure meet HACS requirements)
