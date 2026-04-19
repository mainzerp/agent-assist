# Architecture

## System Overview

agent-assist is a two-component system for natural language smart home control:

1. **Docker Container** -- The AI backend running FastAPI with multi-agent orchestration, a two-tier vector cache, hybrid entity matching, MCP tool integration, and a plugin system.
2. **HA Custom Integration** -- A thin I/O bridge (`custom_components/ha_agenthub/`) that forwards user input to the container and streams responses back to Home Assistant's conversation system.

All configuration, secrets, and state are stored in SQLite. ChromaDB provides vector storage for entity embeddings and cache embeddings. No configuration files are used at runtime -- everything is managed through the setup wizard and admin dashboard.

## Component Diagram

```
+--------------------------------------------------+
|  Home Assistant                                   |
|  +--------------------------------------------+  |
|  |  ha_agenthub custom integration            |  |
|  |  (conversation agent -- thin I/O bridge)   |  |
|  +---------------------+----------------------+  |
+-------------------------|-------------------------+
                          | REST / SSE / WebSocket
                          v
+--------------------------------------------------+
|  Docker Container (FastAPI)                       |
|                                                   |
|  +----------------------------------------------+ |
|  | Setup Wizard / Admin Dashboard               | |
|  +----------------------------------------------+ |
|  | API Layer (conversation, admin, health)       | |
|  +----------------------------------------------+ |
|  | Middleware (auth, tracing, setup redirect)    | |
|  +---+------------------------------------------+ |
|      |                                            |
|  +---v---+   +----------+   +-----------+        |
|  | Orch. |-->| A2A      |-->| Specialist|        |
|  | Agent  |  | Dispatch |   | Agents    |        |
|  +---+---+   +----------+   +-----------+        |
|      |                                            |
|  +---v-----------+   +----------+  +----------+  |
|  | Two-Tier Cache|   | Entity   |  | Presence |  |
|  | (routing +    |   | Matcher  |  | Detector |  |
|  |  response)    |   | (5 sig.) |  +----------+  |
|  +---------------+   +----------+                 |
|                                                   |
|  +---------------+   +----------+  +----------+  |
|  | MCP Tool Mgr  |   | Plugin   |  | LLM      |  |
|  | (stdio/HTTP)  |   | System   |  | Client   |  |
|  +---------------+   +----------+  +----------+  |
|                                                   |
|  +----------------------------------------------+ |
|  | SQLite (config, secrets, history, analytics) | |
|  +----------------------------------------------+ |
|  | ChromaDB (entity index, cache embeddings)    | |
|  +----------------------------------------------+ |
+--------------------------------------------------+
```

## A2A Protocol

Agents communicate via a JSON-RPC 2.0-based Agent-to-Agent (A2A) protocol:

- **Registry** (`a2a/registry.py`) -- Maintains agent cards describing each agent's ID, name, description, skills, and endpoint.
- **Dispatcher** (`a2a/dispatcher.py`) -- Routes JSON-RPC requests to agents by method (`message/send`, `message/stream`, `agent/discover`, `agent/list`).
- **Transport** (`a2a/transport.py`) -- In-process transport calls agent handlers directly within the container. The transport abstraction allows for future HTTP-based transport.

Each agent publishes an **Agent Card** containing its ID, capabilities, and supported intents. The orchestrator uses these cards to make routing decisions.

## Request Flow

1. User speaks a command in Home Assistant (e.g., "turn on the bedroom light").
2. The HA custom integration sends the text to the container via `POST /api/conversation` (or SSE/WebSocket).
3. The API layer authenticates the request (Bearer token) and builds an A2A `message/send` request targeting the orchestrator.
4. **Orchestrator agent** receives the request:
   a. Checks the **routing cache** -- if a similar request was recently routed, reuses the cached routing decision (threshold: 0.92 cosine similarity).
   b. If cache miss, calls the LLM for **intent classification** to select the target agent.
   c. Condenses the task description, preserving entity names.
   d. Dispatches via A2A to the selected specialist agent.
5. **Specialist agent** (e.g., light-agent) receives the task:
   a. Uses the **entity matcher** to resolve "bedroom light" to `light.bedroom_main`.
   b. Calls the HA REST API (`ha_client/rest.py`) to execute `light/turn_on`.
   c. Returns a response with speech text and action details.
6. The orchestrator checks the **response cache** for reuse opportunities and stores the new result.
7. The response flows back through the API layer to the HA integration, which speaks it to the user.

## Two-Tier Cache

The cache system uses ChromaDB vector embeddings for semantic similarity matching:

- **Routing Cache** -- Caches the mapping from user intent to target agent. A hit (cosine similarity >= 0.92) skips LLM-based intent classification entirely. Max entries: 50,000 with LRU eviction.
- **Response Cache** -- Caches full agent responses including executed actions.
  - **Full hit** (>= 0.95): Returns the cached response directly (optionally rewritten by the rewrite agent for variety).
  - **Partial hit** (0.80-0.95): Provides the cached response as context to the agent for faster processing.
  - **Miss** (< 0.80): No cache involvement.
  - Max entries: 20,000 with LRU eviction.

Cache entries are reactively invalidated when an executed action fails.

## Entity Matching

The hybrid entity matcher combines five signals with configurable weights:

| Signal | Method | Example |
|--------|--------|---------|
| Fuzzy string | Levenshtein + Jaro-Winkler | "bedroom lite" ~ "bedroom light" |
| Phonetic | Soundex + Metaphone | "bedroom lite" sounds like "bedroom light" |
| Embedding | ChromaDB vector similarity | Semantic closeness |
| Alias | Exact lookup from DB | "nightstand lamp" = `light.bedroom_nightstand` |
| Domain | HA entity domain filtering | "light" commands only match `light.*` entities |

A weighted score above 0.75 returns a single confident match. Below that threshold, the top-N candidates are sent to the LLM for disambiguation.

## Data Storage

- **SQLite** -- Primary store for all structured data: settings, agent configs, custom agents, aliases, MCP servers, secrets (Fernet-encrypted), admin accounts (bcrypt-hashed), setup state, conversations, analytics, and trace spans.
- **ChromaDB** -- Vector store for entity index embeddings, routing cache embeddings, and response cache embeddings. Persisted to disk at `/data/chromadb`.

## Plugin Architecture

Plugins extend the system without modifying core code:

- Plugins are Python files in `container/plugins/` discovered at startup.
- Each plugin subclasses `BasePlugin` and implements lifecycle hooks: `configure`, `startup`, `ready`, `shutdown`.
- The `PluginContext` provides access to the agent registry, MCP registry, settings repository, and the FastAPI app instance.
- Plugins can register custom A2A agents, add dashboard routes, subscribe to events via the event bus, and read/write settings.
- Plugin failures are isolated -- one plugin crashing does not affect others.

See [Plugin Development Guide](plugin-development.md) for details.
