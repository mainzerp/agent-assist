# Troubleshooting

## Container Won't Start

**Check Docker logs:**

```bash
docker-compose logs agent-assist
```

**Common causes:**

- Port already in use: Change `CONTAINER_PORT` in `.env` or `docker-compose.yml`.
- Volume permission issues: Ensure the Docker volume is writable. On Linux, check that the data directory has appropriate ownership.
- Missing dependencies: Rebuild the image with `docker-compose up -d --build`.
- Python import errors in logs: Indicates a corrupted build. Remove the image and rebuild:
  ```bash
  docker-compose down
  docker rmi agent-assist
  docker-compose up -d --build
  ```

## Can't Connect to Home Assistant

**Verify the HA URL is reachable from inside the container:**

```bash
docker exec agent-assist python -c "
import urllib.request
urllib.request.urlopen('http://<ha_url>:8123/api/')
"
```

**Common causes:**

- Wrong URL: Use the IP address or hostname reachable from the container's network, not `localhost` (unless using host networking).
- Docker networking: If HA is on the same Docker host, use `http://host.docker.internal:8123` (Docker Desktop) or `http://172.17.0.1:8123` (Linux default bridge).
- Invalid token: Generate a new Long-Lived Access Token in HA under Profile > Security.
- Firewall: Ensure port 8123 (or your HA port) is not blocked between the container and HA.

**Re-test from the setup wizard:**

Navigate to `http://<host>:8080/setup/step/2` and use the "Test Connection" button.

## LLM Errors

**Symptoms:** Agent responses contain error messages or the container logs show LLM API errors.

**Common causes:**

- Invalid API key: Re-enter the key in the setup wizard (step 4) or update via the admin dashboard.
- Provider outage: Check the provider's status page (OpenRouter, Groq, Ollama).
- Rate limiting: Reduce request frequency or switch to a different provider/model.
- Ollama not running: If using Ollama, verify the Ollama service is running and accessible at the configured URL.

**Test LLM connectivity:**

Navigate to `http://<host>:8080/setup/` and use the "Test" button for each provider on step 4.

## Entity Not Found

**Symptoms:** Commands like "turn on the bedroom light" return "entity not found" or match the wrong device.

**Steps to resolve:**

1. Check the entity index status page in the admin dashboard (Entity Index page).
2. Trigger a manual refresh: click "Refresh" on the Entity Index dashboard page or call `POST /api/admin/entity-index/refresh`.
3. Verify the entity is exposed in Home Assistant -- only entities visible through the HA REST API (`/api/states`) are indexed.
4. Add an alias: In the admin dashboard, create an alias mapping your preferred name to the exact entity ID (e.g., "bedroom light" -> `light.bedroom_main`).
5. Check entity matching weights: Adjust the signal weights on the Entity Index dashboard page if matches are consistently wrong.

## Cache Not Working

**Symptoms:** Cache hit rate is 0%, or the cache stats page shows no entries.

**Common causes:**

- ChromaDB directory not writable: Check that the volume mount for `/data/chromadb` exists and is writable.
- Embedding engine not initialized: Check container startup logs for embedding-related errors.
- Thresholds too high: Lower the routing cache threshold (default: 0.92) or response cache threshold (default: 0.95) in the admin dashboard.

**Verify ChromaDB:**

```bash
docker exec agent-assist ls -la /data/chromadb
```

## Setup Wizard Issues

**Symptoms:** Stuck on a step, wizard not appearing, or need to redo setup.

**Reset setup state:** Access the SQLite database and clear the setup state:

```bash
docker exec agent-assist python -c "
import sqlite3
conn = sqlite3.connect('/data/agent_assist.db')
conn.execute('DELETE FROM setup_state')
conn.commit()
conn.close()
"
```

Then restart the container:

```bash
docker-compose restart agent-assist
```

## Integration Not Appearing in Home Assistant

**HACS installation:**

1. Verify the repository was added correctly in HACS (Integrations > three-dot menu > Custom repositories).
2. Check that the integration was downloaded and installed (not just added).
3. Restart Home Assistant after installation.

**Manual installation:**

1. Confirm the `custom_components/ha_agenthub/` directory exists in your HA config folder.
2. Check that all files are present: `__init__.py`, `config_flow.py`, `const.py`, `conversation.py`, `manifest.json`, `strings.json`, and `translations/en.json`.
3. Restart Home Assistant.

**After installation:**

1. Go to Settings > Devices & Services > Add Integration.
2. Search for "HA-AgentHub" (integration domain `ha_agenthub`).
3. If it does not appear, check the HA logs for import errors: Settings > System > Logs.

## Slow Responses

**Common causes:**

- LLM provider latency: Check provider response times. Groq is typically faster than OpenRouter for small models. Ollama depends on local hardware.
- Low cache hit rate: Check the cache stats in the admin dashboard. A low hit rate means most requests require LLM calls.
- Agent timeout: Increase agent timeout values in the admin dashboard (Agent Configuration).
- Entity index size: A very large entity index (10,000+ entities) may slow down entity matching. Consider using entity visibility rules to limit which entities each agent can see.

## Log Inspection

**View container logs:**

```bash
docker-compose logs -f agent-assist
```

**Adjust log level:**

Set `LOG_LEVEL=DEBUG` in `docker-compose.yml` or `.env` and restart:

```bash
docker-compose restart agent-assist
```

**Trace IDs:**

Each request is assigned a trace ID (visible in the logs as `[trace:...]`). Use the trace ID to find all related log entries for a single request. Traces can also be viewed in the admin dashboard (Traces page) with a Gantt visualization of each processing step.

**Log format:**

```
2025-01-15 10:30:00 INFO [app.agents.orchestrator] Routing to light-agent
```

Logs include timestamp, level, logger name, and message.
