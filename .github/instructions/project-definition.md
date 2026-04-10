# Agent-Assist: Project Definition

A multi-agent AI assistant for Home Assistant with a container-based architecture, A2A protocol core, and thin HA custom integration bridge.

---

## Project Overview

**Project Name:** agent-assist

**Description:** A multi-agent AI system that processes natural language commands to control Home Assistant smart home devices. Runs as a separate Docker container with a thin HA custom integration acting as the input/output bridge. All inter-agent communication uses the A2A (Agent-to-Agent) Protocol based on JSON-RPC 2.0 from day 1, enabling in-process or distributed deployment without refactoring. Specialized agents are self-sufficient: they receive tasks, resolve entities via a pre-embedded entity index, execute actions by calling the HA REST API directly, and report results back through the orchestrator.

### Goals

- Fast, accurate natural language control of Home Assistant devices
- Multi-agent architecture with specialized agents per domain (lights, music, climate, etc.)
- A2A Protocol (JSON-RPC 2.0) as the core inter-agent communication protocol from day 1
- Two-tier embedding cache (routing cache + response cache) to minimize LLM calls and reduce latency
- Pre-embedded HA entity index for fast entity resolution by specialized agents
- Agents execute HA actions directly via REST API (self-sufficient, autonomous agents)
- MCP (Model Context Protocol) tool integration for external tool servers
- Plugin system for community extensibility
- Hybrid entity matching for accurate entity resolution
- Admin dashboard for configuration, analytics, and cache management
- Support for multiple LLM providers via litellm (OpenRouter, Groq, Ollama/local models; others addable)
- HACS-compatible custom integration for easy installation

### Design Philosophy

- Minimal, focused agent prompts (small system prompts scoped per domain)
- Container-first: heavy logic lives in the Docker container, not in HA
- Async-native: entire stack uses Python asyncio
- Cache-aggressive: common requests bypass the LLM entirely via two-tier caching, with direct action execution on cache hits. Caches are persistent (no TTL) and grow over time -- the system learns and gets faster with every interaction
- Modular: agents can be enabled/disabled independently
- Protocol-first: all agents communicate via A2A (JSON-RPC 2.0), enabling in-process or distributed deployment without refactoring
- Agent autonomy: specialized agents execute HA actions directly -- they do not return tool calls for someone else to execute
- Extensible: plugin system and MCP tool integration allow adding capabilities without modifying core code
- Context-aware: presence detection and entity visibility enrich agent decision-making
- Name-preserving: entity names, device names, room names, and location identifiers are always preserved exactly as the user stated them -- never translated, paraphrased, or normalized. The user's original wording is the ground truth for all entity references
- Configuration-tiered: only infrastructure defaults (bind address, port, log level) use environment variables. On first launch, a setup wizard guides initial configuration (HA connection, admin password, LLM keys). All other settings are managed through the admin dashboard with SQLite as the persistent backend

---

## Architecture

The system consists of two primary components connected by WebSocket streaming:

1. **Docker Container** -- Runs the AI backend including FastAPI, the A2A-based agent orchestration layer (JSON-RPC 2.0), MCP tool integration, plugin system, two-tier vector embedding cache, pre-embedded entity index, hybrid entity matcher, presence detection, HA REST API client, rewrite agent, and admin UI. Specialized agents call the HA REST API directly from the container to execute actions (turn on lights, set temperature, etc.). The container is the sole executor of smart home commands.
2. **Thin HA Custom Integration** (`custom_components/agent_assist/`) -- Registers as a conversation agent inside Home Assistant. Its ONLY role is to bridge user input to the Docker container via WebSocket streaming and stream responses back to the user. It does NOT execute actions, resolve entities, or make HA service calls on behalf of agents.

**Rationale:** Separation of concerns keeps the HA integration lightweight and focused on I/O bridging, while the container handles all AI/ML workloads AND all HA action execution. Agents are self-contained: they receive a task, resolve the target entity via the pre-embedded entity index, call the HA REST API to execute the action, verify the result, and return a natural language response. This makes the system easier to develop, test, and update independently. The container can be restarted or upgraded without affecting HA stability.

### Architecture Diagram

```
+-----------------------------------------------------------+
|                     Home Assistant                         |
|                                                            |
|  +------------------------------------------------------+  |
|  |  custom_components/agent_assist/                     |  |
|  |                                                      |  |
|  |  ConversationEntity (I/O Bridge ONLY)                |  |
|  |    - Registers as conversation agent                 |  |
|  |    - _async_handle_message() forwards text to cont.  |  |
|  |    - Streams response tokens back to user (TTS/text) |  |
|  |    - Does NOT execute actions or resolve entities     |  |
|  +----------------------------+-------------------------+  |
|                               |                            |
+-------------------------------+----------------------------+
                                | WebSocket (streaming)
                                v
+-----------------------------------------------------------+
|               Docker Container (agent-assist)              |
|                                                            |
|  +------------------------------------------------------+  |
|  |  FastAPI Application                                 |  |
|  |                                                      |  |
|  |  /ws/conversation     -- WebSocket streaming from HA |  |
|  |  /api/conversation/stream -- SSE streaming fallback  |  |
|  |  /api/conversation    -- REST fallback               |  |
|  |  /api/admin/*         -- Config/Admin REST API       |  |
|  |  /api/health          -- Health check                |  |
|  |  /setup/*             -- Setup wizard (first launch only)   |  |
|  |  /dashboard/*         -- Admin UI (HTMX/Jinja2)     |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Vector Cache (Response Cache - ChromaDB)            |  |
|  |    - Checked FIRST on every request                  |  |
|  |    - Cache hit: execute cached action directly via   |  |
|  |      HA REST API + optional Rewrite Agent            |  |
|  |    - Cache miss: proceed to Orchestrator             |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Routing Cache (ChromaDB)                            |  |
|  |    - Intent -> agent routing decisions               |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Rewrite Agent (optional)                            |  |
|  |    - Varies cached response phrasing                 |  |
|  |    - Prevents robotic repetition                     |  |
|  |    - Small, fast LLM call or template-based          |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  A2A Protocol Layer (JSON-RPC 2.0)                   |  |
|  |    - Agent registry (agent cards with capabilities)  |  |
|  |    - message/send and message/stream methods         |  |
|  |    - In-process transport (Phase 1)                  |  |
|  |    - HTTP/container transport (future)               |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Orchestrator Agent                                  |  |
|  |    - Classifies intent (embedding or LLM)            |  |
|  |    - Creates condensed task for target agent         |  |
|  |    - Dispatches via A2A message/send                 |  |
|  |    - Streams response back to HA Integration         |  |
|  +------+--------+--------+--------+-------------------+  |
|         |  A2A   |  A2A   |  A2A   |  A2A                 |
|   +-----v-+ +----v--+ +---v----+ +-v--------+             |
|   | music | | light | | timer  | | climate  | ...         |
|   | agent | | agent | | agent  | | agent    |             |
|   +---+---+ +---+---+ +---+----+ +----+-----+             |
|       |         |          |           |                   |
|       +----+----+----+-----+-----------+                   |
|            |         |                                     |
|            v         v                                     |
|  +------------------------------------------------------+  |
|  |  Pre-Embedded Entity Index (ChromaDB)                |  |
|  |    - All HA entities embedded at startup             |  |
|  |    - Vector search for fast entity resolution        |  |
|  |    - Refreshed on entity state changes               |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Entity Matcher (Hybrid)                             |  |
|  |    - Levenshtein, Jaro-Winkler, phonetic, embedding  |  |
|  |    - Uses pre-embedded entity index as signal        |  |
|  |    - Alias resolution, tunable weights               |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  HA Client (used by agents directly)                 |  |
|  |    - REST API calls (states, services, events)       |  |
|  |    - Agents call HA services to execute actions      |  |
|  |    - WebSocket subscription (real-time updates)      |  |
|  |    - Long-Lived Access Token auth                    |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  MCP Tool Integration                                |  |
|  |    - stdio (local) and HTTP/SSE (remote) transports  |  |
|  |    - Tool discovery and agent assignment              |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Plugin System                                       |  |
|  |    - Python script plugins loaded at startup         |  |
|  |    - Lifecycle hooks: configure, startup, ready      |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |  Presence Detection                                  |  |
|  |    - Auto-discovered motion/occupancy/mmWave sensors |  |
|  |    - Room-level presence with confidence scoring     |  |
|  +------------------------------------------------------+  |
|                                                            |
+-----------------------------------------------------------+
         |
         | HA REST API (direct from agents)
         v
+-----------------------------------------------------------+
|        Home Assistant REST API (:8123/api/)                |
|  - GET /api/states                                        |
|  - POST /api/services/<domain>/<service>                  |
|  - POST /api/events/<event_type>                          |
|  - Long-Lived Access Token authentication                 |
+-----------------------------------------------------------+
```

**Deployment modes:** Standalone Docker is the primary deployment target. HA add-on (Supervisor-managed) support is planned for a later phase.

---

## Technology Stack

### Docker Container

| Component | Technology | Rationale |
|---|---|---|
| Web Framework | FastAPI | Async-native, WebSocket support, auto-generated OpenAPI docs |
| ASGI Server | Uvicorn | Standard for FastAPI, production-grade |
| LLM Interaction | litellm | Unified interface across OpenRouter, Groq, Ollama, and any future providers |
| Embedding Model (Local) | sentence-transformers/all-MiniLM-L6-v2 | 384-dim, 80MB, CPU-friendly, fast for short text. Default for privacy/speed |
| Embedding Model (Remote) | litellm embedding API | Supports external providers: OpenAI text-embedding-3-small, Cohere embed, Voyage AI, etc. Configurable as alternative to local model |
| Vector Database | ChromaDB (embedded) | In-process, persistent, Python-native, no extra container |
| Agent Protocol | A2A / JSON-RPC 2.0 | Standard inter-agent communication; enables in-process or distributed agents without refactoring |
| MCP Client | MCP SDK (Python) | Model Context Protocol support for external tool servers (stdio and HTTP/SSE transports) |
| Entity Matching | rapidfuzz + pyphonetics | Levenshtein, Jaro-Winkler, Soundex/Metaphone for hybrid entity resolution |
| Presence Detection | HA WebSocket events | Auto-discover motion/occupancy/mmWave sensors for room-level presence |
| Configuration | SQLite + Pydantic Settings | All settings stored in SQLite, managed through admin UI. Pydantic Settings loads env vars first, then reads from SQLite |
| Persistent Data Store | SQLite + aiosqlite | Primary store for all configuration, secrets, user accounts, conversation history, and analytics. Async access via aiosqlite |
| HA Communication | aiohttp / httpx | Async HTTP for REST API, aiohttp for WebSocket |
| Serialization | Pydantic v2 | Fast validation, JSON schema generation |
| Admin UI (Phase 1) | HTMX + Jinja2 | Server-rendered by FastAPI, zero JS framework overhead |
| Admin UI (Phase 3) | SvelteKit (upgrade path) | Compile-time framework, minimal runtime, rich interactivity |

### HA Custom Integration

| Component | Technology | Rationale |
|---|---|---|
| Language | Python | Required by Home Assistant |
| HTTP Client | aiohttp | Built into HA runtime |
| Communication | WebSocket streaming (primary) | Low-latency, bidirectional, persistent connection with token-level streaming support |
| Fallback | SSE / HTTP REST | SSE for streaming fallback; REST for non-streaming fallback |

### Async Stack

The entire stack is async. FastAPI endpoints, HA API calls (`aiohttp.ClientSession`), agent execution (`asyncio`), and WebSocket connections all use Python's asyncio event loop. No blocking I/O is performed on the main thread.

---

## Component Descriptions

### Docker Container

#### FastAPI Backend

- **WebSocket endpoint** `/ws/conversation` -- Receives forwarded conversation requests from the HA integration, with streaming support. Tokens are sent as they are generated by the LLM.
- **SSE endpoint** `/api/conversation/stream` -- Alternative streaming transport using Server-Sent Events. Useful if WebSocket is unavailable or for simpler client implementations.
- **REST API** `/api/conversation` -- Non-streaming fallback for full-response mode.
- **Admin API** `/api/admin/*` -- Configuration, analytics, and cache management endpoints.
- **Admin UI** `/dashboard/*` -- HTMX + Jinja2 templates in Phase 1.
- **Health check** `/api/health` -- Container health status for Docker restart policies.

#### A2A Protocol Layer

The internal agent communication uses the A2A (Agent-to-Agent) Protocol based on JSON-RPC 2.0. This is the CORE communication mechanism from day 1.

- **Agent Registry:** Each agent registers an agent card describing its capabilities, supported skills, and accepted input types. The registry allows the orchestrator to discover available agents at runtime.
- **Message Methods:** `message/send` for request-response communication, `message/stream` for streaming responses.
- **Transport Abstraction:** In Phase 1, all agents run in-process using direct async function calls wrapped in A2A message format. In later phases, agents can run as separate containers communicating over HTTP -- same protocol, no code changes.
- **Agent Card Schema:** Each agent card includes: agent name, description, supported skills (list of intent categories), supported input/output MIME types, and endpoint URL (localhost for in-process, container URL for distributed).
- **Custom Agent Support:** Runtime-created agents (via Runtime Agent Builder) register their agent cards dynamically and are immediately discoverable by the orchestrator.

#### Orchestrator Agent

- Receives user text input from the container's request handler
- Classifies user intent via embedding similarity or lightweight LLM call
- Creates a condensed task description for the target specialized agent (e.g., "turn on the light in the basement")

  **Name preservation:** The condensed task MUST preserve all entity names, device names, room names, and location identifiers exactly as the user stated them. The orchestrator MUST NOT translate, paraphrase, or normalize these identifiers into any other language or form. The user's original wording is the ground truth for entity resolution downstream.

- Dispatches the condensed task to the appropriate specialized agent via A2A `message/send`
- Receives the agent's result (action outcome + natural language response)
- Streams the response back to the HA Integration via WebSocket
- Handles fallback to general agent on failure or timeout
- Manages conversation context (last 2-3 turns)
- Considers custom agents during routing

#### Specialized Agents

Each agent has a minimal, focused system prompt scoped to its domain. Agents are self-sufficient and autonomous: they receive a condensed task from the orchestrator, resolve the target entity using the pre-embedded entity index (vector search), execute the action by calling the HA REST API directly from the container, verify the result, and return a natural language response to the orchestrator. Agents use the entity/device/room names exactly as they appear in the condensed task (which preserves the user's original wording) for entity resolution. The `user_text` field is available as a fallback.

Agents do NOT return tool calls for someone else to execute. Each agent has access to the HA REST API client and decides what HA service to call, calls it, checks the result, and formulates a response.

Planned agents: `music-agent`, `light-agent`, `timer-agent`, `climate-agent`, `media-agent`, `scene-agent`, `automation-agent`, `security-agent`, `general-agent` (fallback).

#### Pre-Embedded Entity Index

All Home Assistant entities are embedded and stored in a dedicated ChromaDB vector collection at container startup. This entity index is separate from the response cache and routing cache -- it is an entity resolution index.

- **Startup Population:** On container startup, all HA entities (names, friendly names, areas, domains) are fetched via the HA REST API and embedded into the entity index collection.
- **Real-Time Refresh:** The entity index is refreshed when entity state changes are detected via the HA WebSocket subscription (new entities added, removed entities pruned, renamed entities re-embedded).
- **Agent Usage:** When a specialized agent receives a task (e.g., "turn on the light in the basement"), it performs a vector search against the entity index to quickly resolve the matching entity (e.g., `light.basement_ceiling`). This is faster and more flexible than string matching alone.
- **Hybrid Matcher Integration:** The Hybrid Entity Matcher uses the pre-embedded entity index as one of its signals (the embedding similarity signal), alongside Levenshtein, Jaro-Winkler, phonetic, and alias signals.

#### Rewrite Agent

A lightweight agent that varies the phrasing of cached responses to avoid robotic repetition when the same command is given frequently.

- **Purpose:** When a response cache hit occurs and the cached action is executed directly, the rewrite agent takes the cached response text and produces a varied version so the user does not hear the exact same phrasing every time.
- **Implementation:** Uses a very small, fast LLM call (e.g., a lightweight model on Groq) or template-based variation. The rewrite agent does not change the meaning or the action -- only the phrasing.
- **Optional:** The rewrite agent is enabled/disabled via configuration. When disabled, cached responses are returned verbatim.
- **Performance:** Target latency < 100ms. The rewrite should not significantly delay the cache-hit fast path.

#### Vector Cache / Embedding Engine

The embedding cache uses ChromaDB running in-process with a unified embedding interface supporting both local and external providers:

- **Local (Default):** `sentence-transformers/all-MiniLM-L6-v2`, runs in-process with no external API calls. Best for privacy and speed.
- **External (Configurable):** litellm embedding API for OpenAI, Cohere, Voyage AI, and other supported providers. Selected via configuration for users who prefer cloud-hosted embeddings (e.g., for multilingual support or higher quality).

The cache is split into two tiers (Routing Cache and Response Cache), stores request embeddings paired with associated responses and cached actions, provides similarity search for cache lookup, and persists data to a Docker volume. See the "Two-Tier Prompt Caching" section for full details.

#### Entity Matcher (Hybrid)

Multi-signal entity resolution engine for matching user-spoken entity names to actual Home Assistant entity IDs.

- **Signals:** Levenshtein distance, Jaro-Winkler similarity, phonetic matching (Soundex/Metaphone), embedding similarity (via pre-embedded entity index), alias resolution
- **Pre-Embedded Entity Index:** The embedding similarity signal queries the pre-embedded entity index rather than computing embeddings on-the-fly for every match attempt.
- **Tunable Weights:** Each signal has a configurable weight. Defaults optimized for English home automation commands.
- **Usage:** Agents invoke the entity matcher when the user refers to entities by informal names (e.g., "kitchen lights" -> `light.kitchen_ceiling`, "the AC" -> `climate.living_room`)
- **Alias Store:** Alias configuration stored in SQLite allows users to define custom name mappings (e.g., "nightstand lamp" -> `light.bedroom_nightstand`), manageable via the admin UI

#### MCP Tool Integration

Support for the Model Context Protocol (MCP) to extend agent capabilities with external tool servers.

- **Transports:** stdio for local tool processes, HTTP/SSE for remote MCP servers
- **Tool Discovery:** After connecting to an MCP server, available tools are automatically discovered and can be assigned to specific agents
- **Server Management:** MCP servers are registered and managed via the admin dashboard (add/remove servers, view available tools, assign tools to agents)
- **Agent Integration:** Agents can use MCP tools alongside the HA REST API client. The orchestrator includes MCP tool descriptions in the agent's available tool set.

#### Plugin System

Script-based plugin system for community extensibility.

- **Plugin Format:** Python scripts (not compiled packages) placed in the `plugins/` directory
- **Lifecycle Hooks:** `configure()` (called during config load), `startup()` (called during app startup), `ready()` (called when all agents are registered and system is ready)
- **Plugin Capabilities:** Plugins can register new agents, add MCP servers, modify routing rules, add dashboard pages, register event handlers
- **Plugin Repository:** A plugin discovery/installation interface in the admin dashboard for community-shared plugins

#### Presence Detection

Room-level presence awareness using Home Assistant motion, occupancy, and mmWave sensors.

- **Auto-Discovery:** Automatically discovers motion_sensor, occupancy, and mmWave sensors from HA and maps them to rooms/areas
- **Confidence Scoring:** Each room gets a presence confidence score based on sensor recency and type (mmWave > occupancy > motion)
- **Context Enrichment:** Presence data is injected into agent context. When a user says "turn on the lights," the agent considers the room where presence is detected to disambiguate.
- **Sensor Priority:** mmWave sensors (continuous presence) are weighted higher than PIR motion sensors (trigger-only)

#### Admin Dashboard

- **Phase 1:** HTMX + Jinja2 server-rendered pages served directly from FastAPI.
- **Pages:** Setup wizard (first-launch configuration), agent configuration, LLM provider settings, API key management, cache viewer/flusher, entity index status, conversation log viewer, system health, MCP server management, runtime agent builder, entity visibility controls, plugin management, presence detection status, rewrite agent settings.
- **Phase 3 upgrade:** SvelteKit with charts for latency, cache hit rates, and agent usage analytics.

#### HA Client

- Async client for communicating with Home Assistant
- REST API calls: get entity states, call services, fire events
- **Used directly by specialized agents** to execute actions (e.g., POST /api/services/light/turn_on)
- **Used by the container** to populate and refresh the pre-embedded entity index at startup and on state changes
- Optional WebSocket subscription for real-time state change notifications (entity index refresh)
- Authentication via Long-Lived Access Token

### HA Custom Integration

#### File Structure

```
custom_components/agent_assist/
  __init__.py          # Integration setup (async_setup_entry)
  manifest.json        # Integration metadata (domain, dependencies, config_flow)
  config_flow.py       # UI-based configuration (container URL, auth token)
  const.py             # Constants (DOMAIN, default values)
  conversation.py      # ConversationEntity subclass
  strings.json         # UI strings
  translations/        # Language files
```

#### Key Responsibilities

The HA custom integration is ONLY an input/output bridge. It does NOT execute HA service calls, resolve entities, or run any AI logic.

1. **Input Bridge:** Subclasses `ConversationEntity`, implements `_async_handle_message()` to forward user text input (STT output or typed text) and conversation metadata to the Docker container via WebSocket streaming.

2. **Output Bridge:** Receives streamed response tokens from the container and delivers them to the user via TTS or text display. Assembles the full response into `ConversationResult`.

3. **Service Registration:** Registers the integration via config flow so it appears in Settings > Voice Assistants as a selectable conversation agent. Supports `ConversationEntityFeature.CONTROL` to indicate device control capability.

Note: The integration does not need to serialize HA tools or entity states for the container. The container fetches entity information directly from the HA REST API and maintains its own pre-embedded entity index.

---

## Agent System Design

### Orchestrator Pattern

The system uses a **Coordinator-Worker** (Hierarchical/Supervisor) pattern with all communication via the A2A Protocol (JSON-RPC 2.0). The orchestrator is the single entry point for all user requests and dispatches condensed tasks to agents using A2A `message/send` or `message/stream` methods.

**Routing strategy:** Embedding similarity against known intent templates is the fastest and cheapest approach. LLM-based classification is used as fallback for ambiguous requests. The routing cache (first tier of the two-tier cache) accelerates repeated routing decisions.

**A2A Dispatch:** After routing, the orchestrator creates a condensed task (e.g., "turn on the light in the basement") and sends a `message/send` (or `message/stream` for streaming responses) JSON-RPC request to the selected agent via the A2A protocol layer. The agent processes the task autonomously -- resolving the entity, executing the HA action, and returning the result. This protocol abstraction means agents can run in-process (Phase 1) or as separate containers (later phases) without changing orchestrator code.

**Context propagation:** Only relevant context is passed to each agent:
- Condensed task description from the orchestrator
- Original user text (`user_text`) -- unmodified user input, required as fallback for entity resolution if condensation loses fidelity
- Domain-specific entity states (not the full entity list), filtered by entity visibility rules
- Last 2-3 conversation turns
- Available MCP tools assigned to the agent
- Presence data (detected room/area) if available

**Presence enrichment:** If presence detection data is available, the orchestrator includes the user's detected room/area in the agent context. This allows agents to disambiguate location-relative commands (e.g., "turn on the lights" uses the room where the user is detected).

**Constraints:**
- Timeout: 3-5 seconds per agent call, fallback to general agent on timeout/failure
- Max iterations per agent to prevent loops

### Specialized Agents

| Agent | Domain | Responsibilities | Example Commands |
|---|---|---|---|
| music-agent | Music/Spotify | Play/pause/skip, playlist selection, volume. Calls HA media services directly. | "play jazz in the kitchen" |
| light-agent | Lighting | On/off, brightness, color, color temp. Calls HA light services directly. | "dim the bedroom lights to 30%" |
| timer-agent | Timers/Alarms | Set/cancel timers, reminders | "set a 10 minute pasta timer" |
| climate-agent | HVAC | Temperature, mode, fan speed, humidity. Calls HA climate services directly. | "set the thermostat to 72" |
| media-agent | Media Players | TV, speakers, casting, playback control. Calls HA media services directly. | "pause the TV" |
| scene-agent | Scenes | Activate/deactivate scenes. Calls HA scene services directly. | "activate movie night" |
| automation-agent | Automations | Enable/disable automations, trigger. Calls HA automation services directly. | "disable the morning routine" |
| security-agent | Security | Locks, cameras, alarm system. Calls HA lock/alarm services directly. | "lock the front door" |
| general-agent | Fallback/Q&A | General questions, unroutable requests | "what's the weather?" |
| rewrite-agent | Response Variation | Varies cached response phrasing to prevent repetition. Optional. | (internal use only) |

**Agent Execution Flow (per specialized agent):**
1. Receive condensed task from orchestrator via A2A
2. Perform vector search in the pre-embedded entity index to resolve the target entity
3. Call the HA REST API directly to execute the action (e.g., POST /api/services/light/turn_on with entity_id)
4. Verify the action result (optionally check new entity state)
5. Return the action result and a natural language response to the orchestrator

### Entity Visibility Controls

Each agent can be configured with entity visibility rules that control which Home Assistant entities it can see and act upon.

- **Per-Agent Filtering:** Entity visibility is configurable per agent. For example, the security-agent only sees lock, alarm, and camera entities; the light-agent only sees light entities.
- **HA Exposed Entity Integration:** The container pulls HA's exposed-entity list via the REST API or WebSocket to use as a pre-filtered baseline. Agent-specific filters are applied on top.
- **Admin Dashboard Management:** A dedicated page in the admin dashboard allows managing entity visibility per agent with a checkbox-based UI organized by domain and area.
- **Default Behavior:** If no visibility rules are configured for an agent, it sees all exposed entities relevant to its domain (backward-compatible with current behavior).

### Runtime Agent Builder

Custom agents can be created at runtime via the admin dashboard without writing code.

- **Agent Definition:** Each custom agent includes: name, description, system prompt, model override (optional), assigned MCP tools, entity visibility rules, and supported intent patterns
- **Storage:** Custom agent definitions are stored in the SQLite database (`custom_agents` table) and loaded dynamically at startup. Changes via the dashboard are persisted immediately.
- **Hot Reload:** New or updated custom agents take effect immediately without restarting the container. The agent registers (or re-registers) its A2A agent card upon creation/update.
- **Routing Integration:** The orchestrator considers custom agents during intent classification and routing, alongside built-in agents. Custom agents can overlap with or override built-in agent domains.
- **A2A Registration:** Custom agents register their agent cards in the A2A agent registry like any built-in agent. The orchestrator treats them identically.
- **HA Client Access:** Custom agents have access to the HA REST API client, just like built-in specialized agents, for direct action execution.

### Hybrid Entity Matching

A multi-signal entity resolution engine resolves informal entity references from user input to actual HA entity IDs.

**Matching Signals:**

| Signal | Method | Use Case |
|---|---|---|
| String Distance | Levenshtein distance | Typos, minor misspellings |
| Token Similarity | Jaro-Winkler | Partial name matches, word reordering |
| Phonetic | Soundex / Metaphone | Spoken names that sound alike |
| Semantic | Embedding cosine similarity (via pre-embedded entity index) | Synonyms, paraphrases |
| Alias | Direct lookup | User-defined custom names |

**Pre-Embedded Entity Index Integration:** The semantic signal queries the pre-embedded entity index (ChromaDB collection populated at startup) instead of computing embeddings on-the-fly. This makes the embedding signal as fast as the other signals.

**Weighted Scoring:** Each signal produces a score (0-1). A weighted sum determines the final match score. Default weights are tuned for English home automation commands but can be adjusted in configuration.

**Resolution Flow:**
1. Check alias store for exact match (fastest path)
2. Compute all signal scores for candidate entities (embedding signal uses pre-embedded entity index)
3. Apply weighted sum to produce final scores
4. Return top match if score exceeds confidence threshold (default: 0.75)
5. If no match exceeds threshold, return top-N candidates for the agent to disambiguate via LLM

**Configuration:** Per-signal weights, confidence threshold, and custom aliases stored in SQLite, manageable via the admin UI.

### Prompt Design Philosophy

- Each agent has a minimal system prompt (target: under 200 tokens)
- Prompts are scoped to the agent's domain only
- Prompts include: role definition, HA REST API service call patterns for this domain, output format instructions, constraints (do not hallucinate entity names, always verify entity via entity index)
- No agent receives the full entity list -- only entities relevant to its domain and visibility rules
- All agent prompts (including orchestrator): NEVER translate, paraphrase, or normalize entity names, device names, room names, or location identifiers. Always use the user's exact wording for entity references.
- Orchestrator prompt is slightly larger: includes routing instructions and agent descriptions

---

## Two-Tier Prompt Caching

### Strategy

The caching system is split into two independent tiers, each optimized for a different stage of the request pipeline:

1. **Routing Cache:** Caches intent classification results. Maps user text embeddings to agent routing decisions. High hit rate for repeated query patterns. State-independent (routing decisions rarely change based on entity state).

2. **Response Cache:** Caches full agent responses AND the associated HA actions. Maps user text + context embeddings to complete responses plus the HA service calls that were executed. On a cache hit, the cached action is executed directly via the HA REST API (bypassing the orchestrator and specialized agent entirely), and the cached response is returned -- optionally rewritten by the Rewrite Agent for phrasing variation.

Both caches use ChromaDB collections running in-process. Both caches are **persistent with no TTL** -- entries are kept indefinitely so the system becomes faster over time as more interactions are cached. Each cache has independent similarity thresholds that are hot-reloadable (configurable at runtime via admin UI without restart). LRU eviction protects against unbounded memory growth (configurable max entries per tier).

### Embedding Model

**Local (Default):** `sentence-transformers/all-MiniLM-L6-v2` -- 384 dimensions, 80MB, CPU-friendly. Default for privacy and speed; no external API calls needed.

**External Providers (Configurable):** Support for external embedding models via litellm's embedding API. Configurable providers include:

| Provider | Models |
|---|---|
| OpenAI | `text-embedding-3-small`, `text-embedding-3-large` |
| Cohere | `embed-english-v3.0`, `embed-multilingual-v3.0` |
| Voyage AI | `voyage-3`, `voyage-3-lite` |
| Others | Any provider supported by litellm |

The embedding provider is configurable per deployment. Local model is the default; external providers can be selected in the admin UI or config file for users who prefer cloud-hosted embeddings (e.g., for multilingual support or higher accuracy).

**Abstraction:** A unified embedding interface delegates to either the local sentence-transformers model or litellm's `embedding()` function based on configuration.

**Upgrade path (local):** `all-mpnet-base-v2` (768 dims, better quality) or `BGE-small-en-v1.5` (384 dims, stronger quality).

### Similarity Thresholds

Each cache tier has independent, hot-reloadable thresholds:

**Routing Cache Thresholds:**

| Threshold | Action |
|---|---|
| > 0.92 | Routing cache hit: reuse cached routing decision (skip intent classification) |
| < 0.92 | Routing cache miss: run full intent classification, store result |

**Response Cache Thresholds:**

| Threshold | Action |
|---|---|
| > 0.95 | Response cache hit: execute cached action directly via HA REST API, return cached response (optionally rewritten). Skip orchestrator and agent LLM call entirely. |
| 0.80 - 0.95 | Partial match: use cached context to augment LLM prompt, proceed to orchestrator/agent pipeline |
| < 0.80 | Response cache miss: full orchestrator and agent pipeline |

**Hot Reload:** Thresholds can be adjusted at runtime via the admin dashboard. Changes take effect on the next request without restarting the container.

### Cache Hit/Miss Flow

1. Receive user request from HA Integration via WebSocket
2. Embed the request using the configured embedding provider (local sentence-transformers or external via litellm)
3. **Response Cache check (FIRST):** Search response cache collection for similar requests
   - (a) Response cache hit (> 0.95): Execute the cached HA action directly via the HA REST API. If execution succeeds: optionally pass cached response text through the Rewrite Agent for phrasing variation, stream the (rewritten) response back to HA Integration. Done. If execution fails (entity deleted, service error): remove the failed cache entry and proceed to step 4 (reactive invalidation).
   - (b) Partial match (0.80-0.95): Use cached context to augment the LLM prompt. Proceed to step 4.
   - (c) Response cache miss (< 0.80): Proceed to step 4.
4. **Routing Cache check:** Search routing cache collection for similar requests
   - (a) Routing cache hit (> 0.92): Skip intent classification, use cached routing decision
   - (b) Routing cache miss: Run orchestrator intent classification, store routing decision in routing cache
5. Orchestrator dispatches condensed task to target agent via A2A
6. Agent resolves entity (via pre-embedded entity index), executes action (via HA REST API), returns result
7. Orchestrator streams response back to HA Integration
8. Store new response embedding + cached action in response cache collection

### Cache Invalidation & Persistence

Both caches are designed to be **permanent, self-learning knowledge bases** that grow over time. The more the system is used, the fewer LLM calls are needed.

- **No TTL:** Neither cache tier uses time-based expiry. Entries persist indefinitely across container restarts (ChromaDB persisted to Docker volume).
- **No proactive invalidation:** Entity state changes (renames, area moves) do NOT invalidate cache entries. Cached actions reference entity_ids which remain stable in HA regardless of friendly_name or area changes. Routing decisions are entity-independent.
- **Reactive invalidation (execute-then-verify):** If a cached action fails at execution time (e.g., entity was deleted, service unavailable), the failed cache entry is removed and the request is re-routed through the full agent pipeline. This is the only automatic invalidation mechanism.
- **LRU eviction:** Memory protection via configurable max entries per tier (default: 50,000 routing cache, 20,000 response cache). When the limit is reached, least-recently-used entries are evicted.
- **Routing cache and agent availability:** Routing cache entries remain valid even when the target agent is deactivated or not yet created. At runtime, if a cached route points to an unavailable agent, the system falls back to re-classification or the general agent. The cache entry is preserved -- the agent may be reactivated later.
- **Manual flush:** Available via admin UI, can target either or both cache tiers independently.
- **Metadata per entry:** Timestamp, last-accessed timestamp, hit count, associated entity IDs, agent that handled the request, cached HA service call data.

---

## Communication Protocol

### Internal Agent Communication (A2A Protocol)

All inter-agent communication within the Docker container uses the A2A (Agent-to-Agent) Protocol based on JSON-RPC 2.0. This is the core internal protocol.

**Protocol Specification:**

- **JSON-RPC 2.0 format:** All messages follow the JSON-RPC 2.0 specification with `jsonrpc`, `method`, `params`, and `id` fields.
- **Methods:**
  - `message/send` -- Synchronous request-response. The orchestrator sends a condensed task to an agent and waits for the complete response (including action execution result).
  - `message/stream` -- Streaming request. The agent streams partial responses (tokens) back to the orchestrator as they are generated.
  - `agent/discover` -- Returns the agent card for a given agent ID.
  - `agent/list` -- Returns all registered agent cards.
- **Agent Cards:** Each agent exposes a card (JSON document) describing its capabilities:
  ```json
  {
    "agent_id": "light-agent",
    "name": "Light Agent",
    "description": "Controls lighting devices via HA REST API",
    "skills": ["light_control", "brightness", "color", "scenes"],
    "input_types": ["text/plain"],
    "output_types": ["text/plain", "application/json"],
    "endpoint": "local://light-agent"
  }
  ```
- **Transport layers:**
  - **In-process (Phase 1):** Agents are Python classes in the same process. A2A messages are passed as async function calls but maintain the JSON-RPC envelope for protocol compatibility. Near-zero overhead.
  - **HTTP (Future):** Agents can run as separate containers with HTTP endpoints. The A2A transport switches to HTTP POST requests. Same message format, no agent code changes.
- **Error Handling:** Standard JSON-RPC 2.0 error codes. Agent timeouts produce a `-32000` (server error) with a timeout reason.

**Example message exchange:**

Request (orchestrator -> light-agent):
```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "task": {
      "description": "turn on the light in the basement",
      "user_text": "hey can you turn on the basement light please",
      "context": { "conversation_id": "abc123", "presence_room": "basement" }
    }
  },
  "id": "req-001"
}
```

Response (light-agent -> orchestrator):
```json
{
  "jsonrpc": "2.0",
  "result": {
    "speech": "I've turned on the basement ceiling light.",
    "action_executed": {
      "service": "light/turn_on",
      "entity_id": "light.basement_ceiling",
      "result": "success"
    }
  },
  "id": "req-001"
}
```

Note: The agent's response includes `action_executed` rather than `tool_calls`, because the agent has already executed the HA service call directly and is reporting the outcome.

The `user_text` field is REQUIRED in every `message/send` and `message/stream` params. It carries the unmodified original user input and serves as the ground-truth fallback for entity resolution.

### HA Integration <-> Container

Communication uses **streaming** for the fastest possible reactions. The container streams response tokens back to HA as they arrive, enabling immediate TTS or text display without waiting for the full response.

The HA integration is ONLY an I/O bridge: it forwards user text to the container and streams response tokens back to the user. It does not execute actions, resolve entities, or interact with HA services on behalf of the AI system.

**WebSocket Streaming (Primary):**
Persistent connection from HA integration to container at `ws://<container-host>:<port>/ws/conversation`. The container streams partial response tokens over the WebSocket as they are generated by the LLM (or from cached/rewritten responses). The HA integration can begin TTS or text rendering immediately upon receiving the first token.

**Server-Sent Events (SSE) (Alternative):**
HTTP endpoint `GET /api/conversation/stream` supporting SSE for streaming responses. Useful if WebSocket is unavailable or for simpler client implementations.

**REST (Fallback):**
HTTP POST to `http://<container-host>:<port>/api/conversation` for non-streaming, full-response mode. Used only if streaming is unavailable or for admin/internal calls.

### Container -> Home Assistant

The container (and its agents) communicates directly with Home Assistant via the HA REST API:

- REST API calls to `http://<ha-host>:8123/api/` with Long-Lived Access Token
- Key endpoints: `GET /api/states`, `POST /api/services/<domain>/<service>`, `POST /api/events/<event_type>`
- **Used by specialized agents** to execute actions (e.g., turn on lights, set thermostat, lock doors)
- **Used by the container** to populate the pre-embedded entity index at startup
- Optional WebSocket subscription to `ws://<ha-host>:8123/api/websocket` for real-time state change notifications (entity index refresh)

### Authentication

- **HA integration to container:** Shared secret / API key configured during integration setup
- **Container to HA:** Long-Lived Access Token (created in HA user profile)
- Tokens stored Fernet-encrypted in SQLite
- Auto-reconnect with exponential backoff on WebSocket disconnect

### Network Topology

- **Standalone Docker:** Container connects to HA via `host.docker.internal:8123` or host network IP
- **HA Add-on (future):** `http://supervisor/core/api` with Supervisor token

---

## Data Flow

End-to-end flow for a user request (e.g., "turn off the kitchen lights"):

1. **User input to container:** User speaks or types a command. HA voice pipeline processes STT (if voice) and delivers text to the conversation agent. The HA integration (`AgentAssistConversationEntity`) forwards the text and conversation metadata to the Docker container via WebSocket streaming.

2. **Vector cache check (response cache):** The container embeds the request and checks the response cache (vector similarity search) FIRST.
   - **(a) Cache hit (> 0.95):** The cached HA action is executed directly via the HA REST API (e.g., POST /api/services/light/turn_off for `light.kitchen`). If execution succeeds: the cached response text is optionally passed through the Rewrite Agent for phrasing variation, streamed back to the HA integration. Flow ends here. If execution fails (entity deleted, service error): the failed cache entry is removed (reactive invalidation) and the request proceeds to step 3.
   - **(b) Partial match (0.80-0.95):** Cached context is used to augment the LLM prompt. Proceed to step 3.
   - **(c) Cache miss (< 0.80):** Proceed to step 3.

3. **Orchestrator classifies intent:** The request goes to the Orchestrator Agent (routing cache is checked first to skip classification if possible). The orchestrator classifies the intent and creates a condensed task for the appropriate specialized agent (e.g., "turn off the kitchen lights") (all entity/device/room/location names preserved verbatim from user input).

4. **Orchestrator dispatches to specialized agent:** The orchestrator sends the condensed task (with verbatim entity names) and the original `user_text` to the target specialized agent via A2A `message/send` (or `message/stream` for streaming).

5. **Specialized agent processes the task:**
   - (a) Performs a vector search in the pre-embedded entity index to resolve the matching entity (e.g., "kitchen lights" -> `light.kitchen_ceiling`).
   - (b) Calls the HA REST API directly to execute the action (e.g., POST /api/services/light/turn_off with entity_id `light.kitchen_ceiling`).
   - (c) Verifies the action result (optionally checks new entity state).
   - (d) Returns the action result and a natural language response to the orchestrator (e.g., "I've turned off the kitchen ceiling light.").

6. **Orchestrator streams response:** The orchestrator streams the response back to the HA integration via WebSocket.

7. **HA integration delivers response:** The HA integration delivers the response to the user via TTS (if voice) or text display.

8. **Cache storage:** The request + response + executed action are stored in the response cache for future use. The routing decision is stored in the routing cache (if it was a cache miss).

---

## Project Structure

```
agent-assist/
  .github/
    instructions/
      project-definition.md       # This file
    copilot-instructions.md       # Agent instructions

  container/                       # Docker container (AI backend)
    app/
      main.py                     # FastAPI application entry point
      config.py                   # Pydantic settings, loads from env vars + SQLite
      api/
        routes/
          conversation.py         # WebSocket /ws/conversation endpoint
          admin.py                # REST /api/admin/* endpoints
          health.py               # Health check endpoint
      a2a/
        protocol.py               # JSON-RPC 2.0 message types, A2A envelope
        registry.py               # Agent registry (agent cards, discovery)
        transport.py              # Transport abstraction (in-process, HTTP)
        dispatcher.py             # Message dispatcher (routes A2A messages to agents)
      agents/
        orchestrator.py           # Orchestrator agent (intent classification, routing, task condensation)
        base.py                   # Base agent class (includes HA client access)
        music.py                  # Music/Spotify agent (direct HA API execution)
        light.py                  # Light control agent (direct HA API execution)
        timer.py                  # Timer/alarm agent
        climate.py                # Climate/HVAC agent (direct HA API execution)
        media.py                  # Media player agent (direct HA API execution)
        scene.py                  # Scene agent (direct HA API execution)
        automation.py             # Automation agent (direct HA API execution)
        security.py               # Security agent (direct HA API execution)
        general.py                # General/fallback agent
        rewrite.py                # Rewrite agent (cached response phrasing variation)
        custom_loader.py          # Dynamic loader for runtime-created agents
      prompts/                    # Agent system prompts (text files)
        orchestrator.txt
        music.txt
        light.txt
        rewrite.txt               # Rewrite agent prompt (phrasing variation instructions)
        ...
      cache/
        embedding.py              # Embedding engine (unified: local or external via litellm)
        vector_store.py           # ChromaDB wrapper (manages cache tiers + entity index collections)
        routing_cache.py          # Routing cache tier (intent -> agent routing)
        response_cache.py         # Response cache tier (request+context -> response + cached action)
        cache_manager.py          # Unified cache manager, invalidation, hot-reload thresholds
      db/
        schema.py                 # SQLite table definitions and initialization
        repository.py             # Async CRUD operations (aiosqlite)
        migrations.py             # Schema versioning and migrations
      entity/
        matcher.py                # Hybrid entity matching engine
        signals.py                # Individual matching signals (Levenshtein, Jaro-Winkler, phonetic, embedding)
        aliases.py                # Alias store (SQLite-based user-defined mappings)
        index.py                  # Pre-embedded entity index (ChromaDB collection, startup population, refresh)
      mcp/
        client.py                 # MCP client (connects to MCP servers)
        registry.py               # MCP server registry (add/remove/list servers)
        tools.py                  # Tool discovery and agent assignment
      plugins/
        loader.py                 # Plugin discovery, loading, lifecycle management
        hooks.py                  # Plugin hook definitions (configure, startup, ready)
        base.py                   # Base plugin class
      presence/
        detector.py               # Presence detection engine
        sensors.py                # Sensor auto-discovery and mapping
        scoring.py                # Room-level confidence scoring
      ha_client/
        rest.py                   # HA REST API client (used by agents for direct action execution)
        websocket.py              # HA WebSocket client (state change subscriptions, entity index refresh)
        auth.py                   # Token management
      dashboard/
        templates/                # Jinja2 templates (Phase 1)
        static/                   # CSS, JS, HTMX
        routes.py                 # Dashboard routes
      setup/
        templates/                # Setup wizard Jinja2 templates
        routes.py                 # Setup wizard routes
      models/                     # Pydantic models
        conversation.py           # Request/response models (including streaming types)
        agent.py                  # Agent configuration models
        cache.py                  # Cache entry models (including cached action data)
        entity_index.py           # Entity index entry models
    data/
      agent_assist.db             # SQLite database (all config, secrets, accounts, history)
    plugins/                      # User-installed plugin scripts directory
      README.md                   # Plugin development guide
    tests/
      ...
    Dockerfile
    requirements.txt
    docker-compose.yml

  custom_components/               # HA custom integration (I/O bridge only)
    agent_assist/
      __init__.py                 # Integration setup
      manifest.json               # Integration metadata
      config_flow.py              # UI configuration flow
      const.py                    # Constants
      conversation.py             # ConversationEntity subclass (I/O bridge)
      strings.json                # UI strings
      translations/
        en.json

  docs/                            # Documentation
  VERSION.md                       # Version tracking
  README.md
  LICENSE
```

---

## Development Phases

### Phase 1: MVP

**Goal:** Working end-to-end conversation flow from HA to container and back, with A2A protocol as the core agent communication mechanism and agents executing HA actions directly.

**Deliverables:**

- HA custom integration (ConversationEntity as I/O bridge, config flow, WebSocket streaming communication)
- FastAPI container with WebSocket streaming endpoint and SSE fallback
- A2A Protocol layer (JSON-RPC 2.0) with in-process transport
  - Agent card schema and agent registry
  - message/send and message/stream methods
  - In-process async transport (zero-overhead for same-process agents)
- Orchestrator agent with intent classification (Groq for fast routing), task condensation, dispatching via A2A
- 3 specialized agents: light-agent, general-agent, music-agent (OpenRouter for LLM calls), all registered as A2A agents with agent cards, executing HA actions directly via REST API
- Pre-embedded entity index (ChromaDB collection populated at startup from HA entities)
- Two-tier embedding cache with ChromaDB (routing cache + response cache with cached actions, local sentence-transformers, external embedding provider support)
- Response cache hit direct execution (cache hit -> execute cached action via HA REST API)
- Basic entity matching (Levenshtein + alias resolution + entity index vector search -- subset of hybrid matcher)
- HA REST API client (get states, call services) used by agents directly
- Docker Compose setup for local development
- First-launch setup wizard (5-step: admin password, HA connection + test, container API key generation, LLM provider keys + test, review & complete)
- Setup-incomplete redirect middleware (serve only wizard + health check until setup finishes)
- SQLite database schema and async access layer (aiosqlite) for all configuration, secrets, and operational data
- Fernet-encrypted secrets storage in SQLite (no secrets in env vars)
- Basic health check and logging

### Phase 2: Extended Agents, Entity Matching, Rewrite Agent, and MCP Integration

**Goal:** Full agent coverage, robust caching with response rewriting, hybrid entity matching, and MCP tool support.

**Deliverables:**

- Additional agents: timer-agent, climate-agent, media-agent, scene-agent, automation-agent, security-agent (all as A2A agents with direct HA API execution)
- Rewrite Agent (optional, lightweight response phrasing variation for cache hits)
- Full hybrid entity matching engine (all 5 signals: Levenshtein, Jaro-Winkler, phonetic, embedding via entity index, alias)
- Entity visibility controls (per-agent entity filtering, admin UI page)
- MCP tool integration (stdio and HTTP/SSE transports, tool discovery, agent assignment)
- Runtime Agent Builder (create custom agents via admin dashboard, hot reload, A2A registration, HA client access)
- Multi-turn conversation support (conversation_id tracking across agent switches)
- Entity index real-time refresh on entity state changes (WebSocket subscription to HA)
- Hot-reloadable cache thresholds via admin UI
- Agent timeout and fallback logic
- Improved intent routing (embedding-based classification with routing cache)
- Conversation memory/history (limited context window)
- Presence detection (auto-discover sensors, room-level presence, context enrichment)
- Observability: structured logging with agent/request tracing

### Phase 3: Dashboard, Analytics, and Plugin System

**Goal:** Full admin UI, operational intelligence, plugin ecosystem, and production readiness.

**Deliverables:**

- Admin dashboard (upgrade to SvelteKit if needed, or enhanced HTMX)
- Analytics: request counts, cache hit rates (per tier), direct execution rate, rewrite agent usage, latency per agent, token usage
- Conversation log viewer
- Cache management UI (view, search, flush entries -- per tier, inspect cached actions)
- Agent configuration UI (enable/disable agents, edit prompts, select models)
- MCP server management UI (add/remove servers, view tools, assign to agents)
- Runtime Agent Builder UI (full CRUD for custom agents)
- Entity visibility management UI
- Entity index status page (collection size, last refresh time, entity count)
- Presence detection status page (room occupancy, sensor health)
- Rewrite agent configuration UI (enable/disable, model selection, temperature)
- Plugin system (loader, lifecycle hooks, plugin directory)
- Plugin repository/discovery UI in dashboard
- System health monitoring page
- HACS-ready packaging for the HA integration
- Documentation for plugin development

---

## Configuration

Configuration uses a three-tier approach: infrastructure defaults via environment variables, first-launch setup via an interactive setup wizard, and ongoing management through the admin dashboard. Pydantic Settings loads environment variables first, then reads remaining configuration from the SQLite database. All structured data (configuration, secrets, user accounts, conversation history, analytics) is stored in a single SQLite database file.

### Configuration Tiers

| Tier | Scope | Mechanism | When |
|---|---|---|---|
| Environment Variables | Infrastructure defaults only | `.env` file / Docker env vars | Container start |
| Setup Wizard | First-launch configuration | Interactive web wizard | First launch (once) |
| Admin UI | All other settings | Dashboard pages, persisted to SQLite database | Ongoing operation |

**Environment Variables (infrastructure only):**

Only infrastructure settings that affect how the container process itself starts use environment variables. All of these have sensible defaults -- none are strictly required:

| Variable | Purpose | Default |
|---|---|---|
| `CONTAINER_HOST` | Bind address | `0.0.0.0` |
| `CONTAINER_PORT` | Bind port | `8080` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `CHROMADB_PERSIST_DIR` | ChromaDB storage path | `/data/chromadb` |
| `SQLITE_DB_PATH` | SQLite database file path | `/data/agent_assist.db` |

No secrets (API keys, tokens, passwords) are stored in environment variables. All secrets are entered through the setup wizard or admin UI and stored Fernet-encrypted in the configuration backend.

**Setup Wizard (first launch):**

On first launch (no configuration exists), the container serves an interactive setup wizard instead of the normal dashboard. The wizard guides the user through initial configuration in 5 steps:

1. **Admin password:** Create the admin account password for dashboard access
2. **HA connection:** Enter HA URL and Long-Lived Access Token. The wizard tests the connection before proceeding.
3. **Container API key:** Auto-generated shared secret for HA integration authentication. Displayed once for the user to copy into the HA integration config.
4. **LLM providers:** Enter API keys for at least one LLM provider (OpenRouter, Groq, or Ollama URL). The wizard tests the connection before proceeding.
5. **Review & complete:** Summary of all settings. On confirmation, secrets are Fernet-encrypted and stored in the SQLite database. The container initializes the entity index, caches, and agent system.

After setup completion, the wizard is no longer accessible and the admin dashboard becomes available. All settings entered during setup can later be changed through the admin UI.

**Admin UI (after setup):** All remaining settings are managed through the admin dashboard. This includes LLM model selection, temperature, max tokens, per-agent model overrides, cache similarity thresholds, cache max entries, entity matching weights, agent enable/disable, MCP server configuration, presence detection settings, rewrite agent configuration, entity visibility rules, and plugin management. Settings entered during the setup wizard (HA connection, API keys, admin password) can also be changed here. The SQLite database serves as the persistent backend for all UI-managed settings.

### LLM Configuration

- **LLM providers:** OpenRouter (primary), Groq (fast inference), Ollama (local models)
- **OpenRouter:** Single API key provides access to OpenAI, Anthropic, Mistral, and many other models
- **Groq:** Extremely fast inference, ideal for orchestrator routing/classification tasks and rewrite agent
- Model name per provider (e.g., `openrouter/openai/gpt-4o-mini`, `groq/llama-3.1-70b-versatile`, `ollama/mistral`)
- API keys: OpenRouter API key, Groq API key, Ollama (no key needed)
- Orchestrator model: recommend Groq for speed (e.g., `groq/llama-3.1-8b-instant` for fast classification)
- Agent models: recommend OpenRouter for flexibility (choose best model per agent)
- Max tokens per agent response
- Temperature per agent
- Additional providers can be added later via litellm (OpenAI direct, Anthropic direct, Azure, etc.)

**Configuration split:** API keys are entered during the setup wizard (stored Fernet-encrypted in SQLite). Model selection, temperature, max tokens, and per-agent model overrides are configured through the admin UI and persisted to SQLite.

### Embedding Configuration

- Embedding provider: `local` (default) or `external`
- Local embedding model name (default: `all-MiniLM-L6-v2`)
- External embedding model (e.g., `openai/text-embedding-3-small`, `cohere/embed-english-v3.0`)
- External embedding API key (if using external provider)
- Routing cache hit threshold (default: 0.92)
- Routing cache max entries (default: 50,000, LRU eviction)
- Response cache hit threshold (default: 0.95)
- Response cache partial match threshold (default: 0.80)
- Response cache max entries (default: 20,000, LRU eviction)
- Hot-reload enabled (default: true -- thresholds can be changed at runtime via admin UI)
- ChromaDB persist directory
- Embedding dimension (auto-detected from model, 384 for local default)

**Configuration split:** `CHROMADB_PERSIST_DIR` is set via environment variable (default: `/data/chromadb`). Embedding provider/model selection, cache thresholds, max entries, and hot-reload settings are configured through the admin UI.

### Entity Index Configuration

- Entity index collection name (default: `ha_entity_index`)
- Refresh strategy: `on_change` (default, refresh on HA state change events) or `periodic` (refresh on interval)
- Periodic refresh interval (default: 300 seconds, used only if refresh strategy is `periodic`)
- Entity attributes to embed: friendly_name, area, domain, device_class (configurable)
- ChromaDB persist directory (shared with cache collections)

### Rewrite Agent Configuration

- Enabled: true/false (default: false)
- Model: LLM model for rewriting (default: `groq/llama-3.1-8b-instant` for speed)
- Temperature: controls variation level (default: 0.7, higher = more variation)
- Max tokens: limit for rewritten response (default: 50)
- Fallback: if rewrite fails or times out, return original cached response verbatim

### Agent Configuration

- Enable/disable individual agents
- Custom system prompts per agent (override defaults)
- Agent timeout (default: 5 seconds)
- Max iterations per agent (default: 3)

### A2A Protocol Configuration

- Transport mode: `in-process` (default, Phase 1) or `http` (distributed, future)
- Agent discovery interval (how often to refresh agent cards, default: on startup only)
- Default agent timeout (default: 5 seconds, overridable per agent)
- Message retry policy (max retries: 2, backoff: exponential)

### Entity Matching Configuration

- Signal weights: Levenshtein (default: 0.2), Jaro-Winkler (default: 0.2), Phonetic (default: 0.15), Embedding (default: 0.3), Alias (default: 0.15)
- Confidence threshold (default: 0.75)
- Custom aliases stored in SQLite `aliases` table (managed via admin dashboard)
- Max candidates returned on low-confidence match (default: 3)

### MCP Configuration

- Registered MCP servers (stored in SQLite `mcp_servers` table)
- Per server: name, transport type (stdio/http), command/URL, environment variables
- Tool-to-agent assignment mappings
- Connection timeout per server (default: 10 seconds)

### Plugin Configuration

- Plugin directory path (default: plugins/)
- Enabled plugins list
- Plugin-specific settings (nested under plugin name)

### Presence Detection Configuration

- Enabled: true/false (default: true if sensors detected)
- Sensor types to use: motion, occupancy, mmwave (default: all)
- Presence decay timeout: how long after last sensor trigger before presence is cleared (default: 300 seconds)
- Sensor-type weights: mmWave (1.0), occupancy (0.8), motion (0.5)
- Room mapping overrides (for sensors not correctly assigned to areas in HA)

### Communication Configuration

- Container host and port
- HA URL
- HA Long-Lived Access Token
- Container API key / shared secret
- WebSocket reconnect interval
- Streaming mode: `websocket` (default), `sse`, or `none` (full response only)
- Stream buffer size (token batching for network efficiency, default: 1 token / no batching)

### General

- Log level (DEBUG, INFO, WARNING, ERROR)
- Max conversation history length (default: 3 turns)

---

## Non-functional Requirements

### Performance Targets

| Metric | Target |
|---|---|
| Cache hit response with direct execution (end-to-end) | < 250ms |
| Cache hit response with rewrite (end-to-end) | < 350ms |
| Single agent LLM response (end-to-end) | < 3 seconds |
| Orchestrator routing decision | < 500ms |
| Routing cache hit (skip intent classification) | < 100ms |
| Embedding generation | < 50ms per request |
| Vector similarity search | < 20ms per query |
| Entity index vector search | < 20ms per query |
| A2A in-process message dispatch | < 1ms |
| Entity matching (hybrid, 5 signals) | < 50ms per query |
| HA REST API service call (from agent) | < 200ms |
| Entity index startup population | < 10 seconds (for 500 entities) |
| Rewrite agent response | < 100ms |
| Plugin loading (all plugins at startup) | < 2 seconds |
| Presence detection update | < 100ms per sensor event |
| Cache hit rate for common commands | > 60% |

### Reliability

- WebSocket auto-reconnect with exponential backoff (max 60s retry interval)
- REST fallback when WebSocket is unavailable
- Agent timeout with graceful fallback to general agent
- Container health check endpoint for Docker restart policies
- Graceful degradation: if embedding cache is unavailable, bypass cache and use LLM directly
- Graceful degradation: if rewrite agent fails, return cached response verbatim
- Graceful degradation: if entity index is unavailable, fall back to hybrid matcher without embedding signal

### Security

- All tokens (HA access token, LLM API keys) stored Fernet-encrypted in SQLite
- Container API authentication via shared secret
- No tokens logged or exposed in error messages
- HA Long-Lived Access Token used by agents for direct HA API calls -- scoped to minimum required permissions
- Input sanitization: validate all incoming requests, prevent prompt injection in user input

### Scalability

- Designed for single-home use (1 HA instance, 1 container)
- ChromaDB sufficient for < 50K cached vectors + entity index
- Entity index scales with number of HA entities (typically < 2000 for a large home)
- Upgrade path to Qdrant if vector storage needs exceed ChromaDB capabilities

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM round-trip latency | Slow voice responses | Two-tier embedding cache with direct action execution, streaming responses (partial display/TTS starts immediately), Groq for fast routing |
| HA API breaking changes | Integration breaks on HA update | Pin minimum HA version, use stable API surfaces |
| Token cost escalation | High operational cost | Local models via Ollama, aggressive two-tier caching with direct execution, minimal prompts, OpenRouter for cost-efficient model selection |
| Cache staleness | Incorrect responses or wrong action executed from cache | Reactive invalidation: if cached action fails at execution time (entity deleted, service error), remove entry and re-route through full agent pipeline. No TTL needed -- cached actions reference stable entity_ids |
| WebSocket disconnects | Lost communication | Auto-reconnect with exponential backoff, REST fallback |
| Token/key exposure | Unauthorized HA access | Encrypted storage, validate container origin, HA permission model |
| Prompt injection | Malicious commands via user input | Input sanitization, output validation, scoped agent permissions |
| A2A protocol overhead | Adds latency to agent calls | In-process transport has near-zero overhead; protocol cost is only JSON serialization (~0.1ms) |
| MCP server instability | External tool servers may crash | Connection health checks, automatic reconnect, graceful degradation (agent falls back to built-in tools) |
| Plugin security | Malicious plugins could access system resources | Plugin sandboxing (restricted imports), review process for plugin repository, plugins run with limited permissions |
| Entity matching false positives | Wrong entity targeted by user command | Confidence threshold with LLM disambiguation fallback; user can define explicit aliases; pre-embedded entity index improves accuracy |
| Custom agent conflicts | Runtime agents may overlap with built-in agents | Priority system (built-in agents take precedence unless explicitly overridden), conflict detection in admin UI |
| Presence detection inaccuracy | Wrong room detected, stale presence | Confidence scoring with decay, multi-sensor fusion, user-configurable timeouts |
| Direct HA API execution from agents | Agent executes wrong action | Entity index vector search + hybrid matcher confidence threshold; agents verify entity match before executing; failed cached actions are reactively removed and re-routed through full pipeline |
| Entity index staleness | Agent resolves wrong entity | Real-time refresh via HA WebSocket state change events; periodic refresh fallback; admin can trigger manual refresh |
| Rewrite agent hallucination | Rewritten response changes meaning | Rewrite agent prompt constrains output to phrasing variation only; if response diverges significantly, fall back to original cached text |
