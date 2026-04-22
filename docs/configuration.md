# Configuration Reference

## Configuration Tiers

HA-AgentHub uses three configuration tiers:

1. **Environment Variables** -- Infrastructure-only settings required to start the container process. Set in `docker-compose.yml` or a `.env` file.
2. **Setup Wizard** -- One-time configuration for secrets and connections (admin password, HA URL/token, API key, LLM keys). Stored encrypted in SQLite.
3. **Admin Dashboard** -- All runtime settings managed through the web UI at `/dashboard/`. Stored in SQLite and hot-reloadable without restart.

## Environment Variables

These are the only settings that use environment variables. All other configuration is stored in SQLite.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_HOST` | `0.0.0.0` | Host address the server binds to. |
| `CONTAINER_PORT` | `8080` | Port the server listens on. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `CHROMADB_PERSIST_DIR` | `/data/chromadb` | ChromaDB vector store persistence directory. |
| `SQLITE_DB_PATH` | `/data/agent_assist.db` | SQLite database file path. |
| `FERNET_KEY_PATH` | `/data/.fernet_key` | Path to the Fernet encryption key. Backup target. |
| `COOKIE_SECURE` | `false` | Set to `true` when serving the dashboard behind HTTPS so the admin session and CSRF cookies are restricted to TLS. Setting it on plain HTTP silently breaks login (browser drops the cookie). |
| `HF_HUB_OFFLINE` | `0` | When `1`, disables Hugging Face Hub network calls so the local embedding model loads strictly from the cached weights baked into the image. |
| `HA_AGENTHUB_TAG` | `main` | Tag used by `container/docker-compose.yml` when pulling `ghcr.io/mainzerp/ha-agenthub`. Override to pin a release. |

Environment variables are loaded by Pydantic `BaseSettings` in `app/config.py` and support `.env` file loading. `HF_HUB_OFFLINE` and `HA_AGENTHUB_TAG` are read by the compose file rather than by the application.

## SQLite Settings Reference

All runtime settings are stored in the `settings` table, organized by category. These are managed through the admin dashboard and seeded with defaults on first startup.

### Cache Settings

The action cache was named "response cache" in 0.20.x and earlier.
The `cache.response.*` setting keys below are intentionally kept to
preserve user-tuned thresholds across the rename. The export and
import API surface uses the `action` tier name (`response` is still
accepted as a legacy alias on read).

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `cache.routing.threshold` | `0.92` | float | Cosine similarity threshold for routing cache hits |
| `cache.routing.max_entries` | `50000` | int | Maximum routing cache entries (LRU eviction) |
| `cache.response.threshold` | `0.95` | float | Cosine similarity threshold for full action cache hits |
| `cache.response.partial_threshold` | `0.80` | float | Threshold for partial action cache matches |
| `cache.response.max_entries` | `20000` | int | Maximum action cache entries (LRU eviction) |

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

The rewrite agent runs only when `personality.prompt` (see
[Personality](#personality-settings)) is non-empty; the keys below
control model selection and sampling for the rewrite call itself.

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `rewrite.model` | (empty) | string | LLM model used by the rewrite/mediation pass over cached or finalised speech. |
| `rewrite.temperature` | `0.7` | float | Sampling temperature for the rewrite call. |

Managed via `GET/PUT /api/admin/rewrite/config` and the dashboard
"Rewrite" page.

### Communication Settings

These keys remain in the settings table; most live streaming controls
have moved to per-route logic in `container/app/api/routes/conversation.py`
and are no longer the canonical knob. Verify behaviour against the
relevant route before tuning.

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
| `a2a.max_dispatch_timeout` | `60` | int | Hard upper bound (seconds) on a single A2A dispatch, regardless of per-agent overrides. Added in 0.18.31. |
| `agent.dispatch_timeout.<agent_id>` | (unset) | int | Per-agent dispatch timeout override; falls back to the agent's `AgentCard.timeout_sec` and then to `a2a.default_timeout`. Capped by `a2a.max_dispatch_timeout`. Added in 0.18.31. |

### Entity Sync Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `entity_sync.interval_minutes` | `30` | int | Background entity-index resync interval. `0` disables periodic syncing (entity index is still primed at startup). |

### Filler Agent Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `filler.enabled` | `false` | bool | Emit interim TTS "thinking" tokens (`StreamToken.is_filler=true`) while the real answer is being generated. |
| `filler.threshold_ms` | `1000` | int | Minimum elapsed milliseconds before the filler agent is allowed to emit. |

### Mediation Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `mediation.model` | (empty) | string | LLM model used by the mediation pass. Empty disables mediation. |
| `mediation.temperature` | `0.3` | float | Sampling temperature for the mediation call. |
| `mediation.max_tokens` | `2048` | int | Maximum tokens for the mediation reply. |

### Personality Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `personality.prompt` | (empty) | string | Personality system prompt for the response mediation/rewrite pass. When non-empty, the rewrite agent is enabled and finalised speech is run through the mediation pipeline. |

Managed via `GET/PUT /api/admin/personality/config` and the dashboard
"Personality" page.

### Language Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `language` | `auto` | string | Conversation language. `auto` enables per-turn language detection; an ISO code (`en`, `de`, ...) pins all replies. The detected/forced language is injected into each agent's system prompt as a directive (see `container/app/agents/actionable.py`). |

### Home Context Settings

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `home.timezone` | (empty) | string | Override timezone for time/date references in agent prompts. Empty falls back to HA's configured timezone. |
| `home.location_name` | (empty) | string | Friendly home name surfaced in prompts and the personality pipeline. |

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
| `reasoning_effort` | (empty) | Optional reasoning-effort hint forwarded to providers that accept it (`Low`, `Medium`, `High`). Added in 0.11.0. |

Default routable agents: `orchestrator`, `light-agent`, `music-agent`,
`timer-agent`, `climate-agent`, `media-agent`, `scene-agent`,
`automation-agent`, `security-agent`, `general-agent`, `send-agent`
(added in 0.12.0; delivers messages to phones, satellites, and
notification targets).

The orchestrator uses a lower temperature (0.3) for consistent intent classification.

Internal helper agents (`filler-agent`, `rewrite-agent`,
`mediation`, `notification-dispatcher`, `cancel-speech`,
`language-detect`, `sanitize`, `delayed-tasks`, `alarm-monitor`,
`custom-loader`) participate in the pipeline but are not user-routable
intent targets and are not listed in the dashboard's intent-routing
configuration.

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

The routing cache and action cache use separate ChromaDB collections. Each entry stores the text embedding, metadata (agent ID, timestamp, hit count), and the cached decision or response.

Cache can be flushed per tier or entirely from the admin dashboard or via the API (`POST /api/admin/cache/flush`).

## MCP Server Configuration

MCP (Model Context Protocol) servers are managed through the admin dashboard or API (`/api/admin/mcp-servers`):

- **Transport**: `stdio` (subprocess) or `sse` (HTTP/Server-Sent Events). The legacy `http` alias was removed in 0.13.0; see `VERSION.md`.
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
