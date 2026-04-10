# Configuration Reference

## Configuration Tiers

agent-assist uses three configuration tiers:

1. **Environment Variables** -- Infrastructure-only settings required to start the container process. Set in `docker-compose.yml` or a `.env` file.
2. **Setup Wizard** -- One-time configuration for secrets and connections (admin password, HA URL/token, API key, LLM keys). Stored encrypted in SQLite.
3. **Admin Dashboard** -- All runtime settings managed through the web UI at `/dashboard/`. Stored in SQLite and hot-reloadable without restart.

## Environment Variables

These are the only settings that use environment variables. All other configuration is stored in SQLite.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_HOST` | `0.0.0.0` | Host address the server binds to |
| `CONTAINER_PORT` | `8080` | Port the server listens on |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CHROMADB_PERSIST_DIR` | `/data/chromadb` | ChromaDB vector store persistence directory |
| `SQLITE_DB_PATH` | `/data/agent_assist.db` | SQLite database file path |

Environment variables are loaded by Pydantic `BaseSettings` in `app/config.py` and support `.env` file loading.

## SQLite Settings Reference

All runtime settings are stored in the `settings` table, organized by category. These are managed through the admin dashboard and seeded with defaults on first startup.

### Cache Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `cache.routing.threshold` | `0.92` | float | Cosine similarity threshold for routing cache hits |
| `cache.routing.max_entries` | `50000` | int | Maximum routing cache entries (LRU eviction) |
| `cache.response.threshold` | `0.95` | float | Cosine similarity threshold for full response cache hits |
| `cache.response.partial_threshold` | `0.80` | float | Threshold for partial response cache matches |
| `cache.response.max_entries` | `20000` | int | Maximum response cache entries (LRU eviction) |

### Embedding Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `embedding.provider` | `local` | string | Embedding provider: `local` or `external` |
| `embedding.local_model` | `all-MiniLM-L6-v2` | string | Local embedding model name |
| `embedding.external_model` | (empty) | string | External model (e.g., `openai/text-embedding-3-small`) |
| `embedding.dimension` | `384` | int | Embedding vector dimension |

### Entity Matching Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `entity_matching.confidence_threshold` | `0.75` | float | Minimum confidence score for entity match |
| `entity_matching.top_n_candidates` | `3` | int | Number of candidates for LLM disambiguation |

### Presence Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `presence.enabled` | `true` | bool | Enable presence detection |
| `presence.decay_timeout` | `300` | int | Seconds before presence confidence decays |

### Rewrite Agent Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `rewrite.enabled` | `false` | bool | Enable rewrite agent for cached response variation |
| `rewrite.model` | `groq/llama-3.1-8b-instant` | string | LLM model for rewrite agent |
| `rewrite.temperature` | `0.8` | float | Temperature for rewrite generation |

### Communication Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `communication.streaming_mode` | `websocket` | string | Streaming mode: `websocket`, `sse`, or `none` |
| `communication.ws_reconnect_interval` | `5` | int | WebSocket reconnect interval in seconds |
| `communication.stream_buffer_size` | `1` | int | Token batching buffer size |

### A2A Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `a2a.default_timeout` | `5` | int | Default agent timeout in seconds |
| `a2a.max_iterations` | `3` | int | Max iterations per agent to prevent loops |

### General Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `general.conversation_context_turns` | `3` | int | Number of conversation turns to keep as context |

## Agent Configuration

Each agent has per-agent settings stored in the `agent_configs` table:

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `1` | Whether the agent is active |
| `model` | Varies per agent | LLM model identifier (e.g., `groq/llama-3.1-8b-instant`, `openrouter/openai/gpt-4o-mini`) |
| `timeout` | `5` | Maximum response time in seconds |
| `max_iterations` | `3` | Maximum processing iterations |
| `temperature` | `0.7` | LLM sampling temperature |
| `max_tokens` | `256` | Maximum tokens per LLM response |

Default agents: `orchestrator`, `light-agent`, `music-agent`, `timer-agent`, `climate-agent`, `media-agent`, `scene-agent`, `automation-agent`, `security-agent`, `general-agent`.

The orchestrator uses a lower temperature (0.3) for consistent intent classification.

## Entity Matching Configuration

Entity matching signal weights are stored in the `entity_matching_config` table and can be adjusted from the admin dashboard:

- **Levenshtein distance** -- Fuzzy string matching
- **Jaro-Winkler similarity** -- String similarity favoring prefix matches
- **Phonetic matching** -- Soundex/Metaphone for sound-alike names
- **Embedding similarity** -- ChromaDB vector cosine similarity
- **Alias lookup** -- Exact match from the aliases table

Aliases are managed in the `aliases` table and can be created/deleted from the admin dashboard. Example: alias "nightstand lamp" resolves to `light.bedroom_nightstand`.

## Cache Configuration

Cache thresholds and max entries are managed as SQLite settings (see Cache Settings above). Changes take effect immediately without restart.

The routing cache and response cache use separate ChromaDB collections. Each entry stores the text embedding, metadata (agent ID, timestamp, hit count), and the cached decision or response.

Cache can be flushed per tier or entirely from the admin dashboard or via the API (`POST /api/admin/cache/flush`).

## MCP Server Configuration

MCP (Model Context Protocol) servers are managed through the admin dashboard or API (`/api/admin/mcp-servers`):

- **Transport**: `stdio` (subprocess) or `http` (HTTP/SSE)
- **Command/URL**: The subprocess command or server URL
- **Environment variables**: Key-value pairs passed to the subprocess
- **Timeout**: Connection timeout in seconds (default: 30)

MCP tools are discovered automatically after connection and can be assigned to specific agents.

## Security Configuration

- **API Key**: Generated during setup, used for HA integration <-> container authentication. Stored Fernet-encrypted in the `secrets` table.
- **Admin Accounts**: Username + bcrypt-hashed password in the `admin_accounts` table. Session-based authentication for the dashboard using signed cookies.
- **HA Token**: Long-Lived Access Token stored Fernet-encrypted in the `secrets` table.
- **LLM API Keys**: Stored Fernet-encrypted in the `secrets` table.
- **Input Sanitization**: User input is sanitized against prompt injection patterns before processing.
