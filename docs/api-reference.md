# API Reference

## Authentication

All API endpoints (except `/api/health` and `/setup/*`) require authentication.

### Conversation Endpoints

Use a Bearer token in the `Authorization` header:

```
Authorization: Bearer <api_key>
```

The API key is generated during the setup wizard (step 3).

### Admin Endpoints

Admin endpoints require a session cookie obtained by logging in through the dashboard. The cookie name is `agent_assist_session` and expires after 24 hours.

### WebSocket

Pass the API key in the `Authorization` header (recommended):

```
Authorization: Bearer <api_key>
```

**Deprecated:** The `token` query parameter is still accepted but will be removed in a future release. Query-string credentials can leak through proxy logs and browser history. Migrate to header-based auth.

---

## Health

### GET /api/health

Returns container health status. No authentication required.

**Response:**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "log_level": "INFO"
}
```

---

## Conversation

### POST /api/conversation

Send a natural language command and receive a full response.

**Auth:** Bearer token

**Request body:**

```json
{
  "text": "turn on the bedroom light",
  "conversation_id": "optional-conversation-id"
}
```

**Response:**

```json
{
  "speech": "I've turned on the bedroom light.",
  "conversation_id": "abc123"
}
```

### POST /api/conversation/stream

Send a command and receive a streaming SSE response.

**Auth:** Bearer token

**Request body:** Same as `POST /api/conversation`

**Response:** Server-Sent Events stream. Each event:

```
data: {"token": "I've", "done": false, "conversation_id": null}
data: {"token": " turned on", "done": false, "conversation_id": null}
data: {"token": "", "done": true, "conversation_id": "abc123"}
```

### WS /ws/conversation

WebSocket endpoint for streaming conversation.

**Auth:** Bearer token via `Authorization` header (preferred). Query-string `token` parameter is deprecated.

**Send:**

```json
{
  "text": "turn on the bedroom light",
  "conversation_id": "optional-id"
}
```

**Receive:** Stream of token objects, same format as SSE events.

---

## Admin -- Settings

### GET /api/admin/settings

Get all settings grouped by category.

**Auth:** Admin session

**Response:**

```json
{
  "settings": {
    "cache": [
      {"key": "cache.routing.threshold", "value": "0.92", "value_type": "float", "category": "cache", "description": "..."}
    ],
    "embedding": [...],
    "entity_matching": [...]
  }
}
```

### PUT /api/admin/settings

Update multiple settings.

**Auth:** Admin session

**Request body:**

```json
{
  "cache.routing.threshold": "0.90",
  "presence.enabled": "true"
}
```

### PUT /api/admin/settings/{key}

Update a single setting.

**Auth:** Admin session

**Request body:**

```json
{
  "value": "0.90",
  "value_type": "float",
  "category": "cache"
}
```

---

## Admin -- Agents

### GET /api/admin/agents

List all registered agents with their configuration.

**Auth:** Admin session

**Response:**

```json
{
  "agents": [
    {
      "agent_id": "light-agent",
      "name": "Light",
      "description": "Lighting control",
      "enabled": true,
      "model": "openrouter/openai/gpt-4o-mini",
      "timeout": 5,
      "temperature": 0.7,
      "max_tokens": 256
    }
  ]
}
```

---

## Admin -- Custom Agents

### GET /api/admin/custom-agents

List all custom agents.

### POST /api/admin/custom-agents

Create a custom agent.

**Request body:**

```json
{
  "name": "weather-agent",
  "description": "Weather information",
  "system_prompt": "You are a weather assistant...",
  "model_override": "openrouter/openai/gpt-4o-mini",
  "intent_patterns": ["weather", "forecast", "temperature outside"]
}
```

### GET /api/admin/custom-agents/{name}

Get a single custom agent.

### PUT /api/admin/custom-agents/{name}

Update a custom agent. Partial updates supported (only include fields to change).

### DELETE /api/admin/custom-agents/{name}

Delete a custom agent.

---

## Admin -- MCP Servers

### GET /api/admin/mcp-servers

List all MCP servers with connection status.

### POST /api/admin/mcp-servers

Add a new MCP server.

**Request body:**

```json
{
  "name": "my-tools",
  "transport": "stdio",
  "command_or_url": "python my_mcp_server.py",
  "env_vars": {"API_KEY": "..."},
  "timeout": 30
}
```

### DELETE /api/admin/mcp-servers/{name}

Remove an MCP server.

### GET /api/admin/mcp-servers/{name}/tools

List discovered tools for a specific MCP server.

---

## Admin -- Entity Index

### GET /api/admin/entity-index/stats

Get entity index statistics with per-domain breakdown.

**Response:**

```json
{
  "count": 150,
  "last_refresh": "2025-01-15T10:30:00",
  "domains": {"light": 45, "switch": 30, "climate": 5}
}
```

### POST /api/admin/entity-index/refresh

Force a full entity index refresh from Home Assistant.

---

## Admin -- Entity Visibility

### GET /api/admin/entity-visibility/{agent_id}

Get visibility rules for an agent.

### PUT /api/admin/entity-visibility/{agent_id}

Set visibility rules for an agent.

**Request body:**

```json
{
  "rules": [
    {"rule_type": "domain", "rule_value": "light"},
    {"rule_type": "area", "rule_value": "bedroom"}
  ]
}
```

### GET /api/admin/entities

List all Home Assistant entities grouped by domain and area.

---

## Admin -- Cache

### GET /api/admin/cache/stats

Get cache statistics per tier.

### GET /api/admin/cache/entries

Browse/search cache entries.

**Query parameters:**
- `tier` -- `routing` or `response` (default: `routing`)
- `search` -- Text filter
- `page` -- Page number (default: 1)
- `per_page` -- Results per page (default: 50, max: 200)

### POST /api/admin/cache/flush

Flush cache entries.

**Request body:**

```json
{
  "tier": "routing"
}
```

Omit `tier` or set to `null` to flush all tiers.

---

## Admin -- Conversations

### GET /api/admin/conversations

List/search conversation history.

**Query parameters:**
- `agent_id` -- Filter by agent
- `search` -- Text search
- `start_date`, `end_date` -- Date range filter
- `page`, `per_page` -- Pagination

### GET /api/admin/conversations/{conversation_id}

Get full thread detail for a conversation.

---

## Admin -- Analytics

### GET /api/admin/analytics/overview

Summary metrics (total requests, avg latency, cache hit rate, total conversations).

**Query parameters:**
- `hours` -- Time window (default: 24, max: 720)

### GET /api/admin/analytics/requests

Time-series request counts in Chart.js-compatible format.

**Query parameters:**
- `hours` -- Time window
- `bucket_minutes` -- Bucket size (default: 60)

---

## Admin -- Traces

### GET /api/admin/traces

List recent traces with pagination.

**Query parameters:**
- `page`, `per_page` -- Pagination

### GET /api/admin/traces/{trace_id}

Get all spans for a specific trace (for Gantt visualization).

---

## Admin -- Presence

### GET /api/admin/presence/status

Get current room confidence scores, sensors, and configuration.

### PUT /api/admin/presence/config

Update presence detection settings.

**Request body:**

```json
{
  "enabled": true,
  "decay_timeout": 300
}
```

---

## Admin -- Plugins

### GET /api/admin/plugins

List all installed plugins with loaded status.

### POST /api/admin/plugins/{name}/enable

Enable a plugin.

### POST /api/admin/plugins/{name}/disable

Disable a plugin.

---

## Setup Wizard

The setup wizard endpoints are used during initial configuration. They are not intended for external API use.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/setup/` | Redirect to first incomplete step |
| GET | `/setup/step/{n}` | Render step template |
| POST | `/setup/step/1` | Save admin password |
| POST | `/setup/step/2` | Save HA connection |
| POST | `/setup/step/3` | Generate API key |
| POST | `/setup/step/4` | Save LLM provider keys |
| POST | `/setup/step/5` | Complete setup |
| POST | `/setup/test/ha` | Test HA connection |
| POST | `/setup/test/llm` | Test LLM provider |
