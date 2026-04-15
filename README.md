# agent-assist

A multi-agent AI assistant for Home Assistant with container-based A2A orchestration, two-tier vector caching, hybrid entity matching, MCP tool integration, and a plugin system.

![Test](https://github.com/<owner>/agent-assist/actions/workflows/test.yml/badge.svg?branch=main)
![Lint](https://github.com/<owner>/agent-assist/actions/workflows/lint.yml/badge.svg?branch=main)
![Docker Build](https://github.com/<owner>/agent-assist/actions/workflows/docker-build.yml/badge.svg?branch=main)
![HACS Validation](https://github.com/<owner>/agent-assist/actions/workflows/hacs-validation.yml/badge.svg?branch=main)

## Features

- **Multi-agent orchestration** -- Specialized agents for lights, music, climate, media, timers, scenes, automation, and security, coordinated by a central orchestrator via the A2A protocol
- **A2A protocol** -- JSON-RPC 2.0-based agent-to-agent communication with registry, dispatcher, and in-process transport
- **Two-tier vector cache** -- Routing cache (skip intent classification) and response cache (skip entire agent pipeline) using ChromaDB embeddings with configurable similarity thresholds
- **Hybrid entity matching** -- Five-signal weighted matcher (Levenshtein, Jaro-Winkler, phonetic, embedding similarity, alias lookup) with LLM disambiguation fallback
- **MCP tool integration** -- Connect external tool servers via Model Context Protocol (stdio and SSE transports) and assign tools to agents
- **Plugin system** -- Extend functionality with Python plugins that register agents, subscribe to events, add dashboard routes, and access settings
- **Admin dashboard** -- 14-page HTMX-powered dashboard for managing agents, entities, cache, MCP servers, analytics, traces, presence, and plugins
- **Presence detection** -- Room-level presence awareness using motion, occupancy, and mmWave sensors with weighted scoring and decay
- **Custom agents** -- Create LLM-powered agents via the dashboard with custom system prompts, model selection, MCP tools, and intent patterns
- **Rewrite agent** -- Optional response variation for cached responses to avoid repetitive answers
- **Setup wizard** -- Guided 5-step first-launch configuration (admin account, HA connection, API key, LLM providers, review)
- **Analytics and tracing** -- Request counts, cache hit rates, latency tracking, token usage, and per-request trace span Gantt visualization

## Architecture

agent-assist runs as a Docker container with a FastAPI backend. A thin Home Assistant custom integration (`custom_components/agent_assist/`) acts as the I/O bridge, forwarding user commands to the container and streaming responses back.

All configuration, secrets, and state are stored in SQLite. ChromaDB provides vector storage for entity embeddings and cache embeddings. No configuration files are used at runtime.

See [docs/architecture.md](docs/architecture.md) for component diagrams, request flow, and detailed design.

## Quick Start

### Prerequisites

- Docker Engine 20.10+ and Docker Compose v2
- A running Home Assistant instance (2024.1.0+)
- An LLM API key (OpenRouter, Groq, or Ollama)

### 1. Clone and Start

```bash
git clone https://github.com/<owner>/agent-assist.git
cd agent-assist/container
docker-compose up -d
```

### 2. Run the Setup Wizard

Open `http://<docker-host>:8080/setup/` in your browser and follow the 5-step wizard:

1. Create an admin account
2. Connect to Home Assistant (URL + Long-Lived Access Token)
3. Generate a container API key (save it -- shown once)
4. Configure LLM provider(s)
5. Review and complete

### 3. Install the HA Integration

**Via HACS (recommended):**

1. In HACS, add `https://github.com/<owner>/agent-assist` as a custom repository (category: Integration).
2. Install "Agent Assist" and restart Home Assistant.

**Manual:**

Copy `custom_components/agent_assist/` to your HA `config/custom_components/` directory and restart HA.

**Configure:**

In HA, go to Settings > Devices & Services > Add Integration > "Agent Assist". Enter the container URL and API key.

## Configuration

agent-assist uses three configuration tiers:

1. **Environment variables** -- Infrastructure-only (`CONTAINER_PORT`, `LOG_LEVEL`, etc.), set in `docker-compose.yml`
2. **Setup wizard** -- One-time secrets and connections, stored encrypted in SQLite
3. **Admin dashboard** -- All runtime settings, hot-reloadable without restart

See [docs/configuration.md](docs/configuration.md) for the full reference.

## Documentation

- [Deployment Guide](docs/deployment.md) -- Docker setup, setup wizard, HA integration, networking, backup
- [Configuration Reference](docs/configuration.md) -- Environment variables, SQLite settings, agent config
- [Architecture Overview](docs/architecture.md) -- Components, A2A protocol, request flow, cache, entity matching
- [API Reference](docs/api-reference.md) -- All REST, SSE, and WebSocket endpoints
- [Plugin Development](docs/plugin-development.md) -- Writing plugins, lifecycle hooks, event bus
- [Troubleshooting](docs/troubleshooting.md) -- Common issues and solutions

## Development

### Run Tests

```bash
cd container
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

### Lint

```bash
cd container
ruff check .
ruff format --check .
```

### Project Structure

```
container/          Docker container (FastAPI backend)
  app/              Application code
    agents/         Specialized agents + orchestrator
    a2a/            A2A protocol (registry, dispatcher, transport)
    api/routes/     REST/SSE/WebSocket endpoints
    cache/          Two-tier vector cache
    dashboard/      Admin dashboard (HTMX + Jinja2 templates)
    db/             SQLite schema + repository
    entity/         Hybrid entity matcher
    ha_client/      Home Assistant REST + WebSocket client
    llm/            LLM client (litellm)
    mcp/            MCP tool integration
    middleware/     Auth + tracing middleware
    models/         Pydantic models
    plugins/        Plugin system
    presence/       Presence detection
    security/       Encryption, hashing, sanitization
    setup/          Setup wizard
  plugins/          User plugins directory
  tests/            Test suite
custom_components/  HA custom integration
  agent_assist/     Thin I/O bridge
```

## Plugin Development

Plugins extend agent-assist without modifying core code. Create a `.py` file in `container/plugins/`, subclass `BasePlugin`, and implement lifecycle hooks.

See [docs/plugin-development.md](docs/plugin-development.md) for the full guide.

## License

MIT
