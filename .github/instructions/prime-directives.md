---
applyTo: "**"
---

# Prime Directives

These are project-critical rules for HA-AgentHub. They describe constraints that, if violated, will break the system's correctness, reliability, or architecture. Always verify new code and changes against these before implementing.

---

## 1. Container is the Execution Engine

All orchestration, entity resolution, LLM calls, Home Assistant action execution, caching, and agent dispatch happen inside the container. The Home Assistant integration is an I/O bridge only — it forwards turns and streams responses. It must never resolve entities, classify intent, or call Home Assistant services on behalf of agents.

## 2. Action Cache Hits Require a Visibility Recheck

An action-cache hit must never replay a cached Home Assistant action without first rechecking entity visibility for that turn. If the recheck fails, the entry is invalidated and the turn falls through to live orchestration. Skipping the recheck is a correctness violation.

## 3. Agents Return Executed Results, Not Tool-Call Plans

Domain agents (light-agent, climate-agent, etc.) are responsible for resolving the target entity, calling the relevant Home Assistant service from the container, and returning a verified execution result. Returning a speculative tool plan or a placeholder response without executing the action is not acceptable.

## 4. Entity Resolution Uses the Deterministic Pipeline First

Entity resolution must always attempt exact `entity_id` lookup and exact friendly-name lookup before falling back to hybrid scoring. Hybrid matching (fuzzy, embedding, phonetic) is a fallback, not the default path. Verbatim user terms must be tried before translated or condensed text.

## 5. Visibility Rules Are Applied on Every Resolution Path

Entity visibility rules (domain, area, entity, device-class filters) must be applied during deterministic lookup, hybrid matching, and cached-action visibility rechecks. Bypassing visibility at any stage leaks entities that should not be accessible to a given agent.

## 6. A2A Protocol Boundary Is Not Bypassed

Agent-to-agent communication goes through A2A message envelopes shaped around JSON-RPC 2.0 semantics. Agents must not call each other directly outside this boundary. The orchestrator dispatches through the A2A layer even for in-process transport.

## 7. The Routing Cache Never Skips Visibility

The routing cache stores query-to-agent routing decisions. A routing-cache hit may skip reclassification, but it must not skip entity visibility enforcement or action-cache replay validation. The live execution path must still be used when a routing hit does not have a safe action-cache entry.

## 8. Plugins and Custom Agents Are Trusted but Scoped

Plugins run in-process and are treated as trusted code. They must not bypass the A2A dispatch layer, must not call Home Assistant services directly without going through the container's action execution path, and must not register conflicting agent IDs. Custom agents loaded from the database follow the same agent contract as built-in agents.

## 9. Async All the Way Down

No blocking I/O is permitted on the asyncio event loop. All Home Assistant REST and WebSocket calls, cache operations, database access, entity-index operations, and agent dispatch are async. Blocking or CPU-intensive operations must be pushed off the event loop.

## 10. Current State Only in Project Docs

Documentation, code comments, and analysis must describe the current runtime state of the project. Roadmap items must be clearly labeled as such and kept in `docs/roadmap.md`. Never describe planned or aspirational behavior as if it were already implemented.
