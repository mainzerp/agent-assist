# HA snapshot fixtures

These JSON files are hand-curated fixtures shaped to match Home Assistant's
REST API responses (`/api/states`, `/api/config`) plus a couple of helper
maps (areas, devices) consumed by the real-scenario test runner.

**No PII, ever.** Every name, area, and device id is fictional.

## Files

- `home_default.json` — list of entity state dicts. Shape mirrors
  `GET /api/states` exactly so `entity.ingest.parse_ha_states` consumes
  it without modification.
- `home_default.areas.json` — `area_id -> {name, name_de}` map. Used by
  the runner to seed area metadata on the recording HA client.
- `home_default.devices.json` — `device_id -> {area_id, name}` map.
  Three `assist_satellite.*` devices expose stable area / device ids
  the timer/light/etc. scenarios reference.
- `home_default.config.json` — payload returned by the recording client's
  `get_config()`; feeds `home_context_provider` (timezone, location).

## Provenance

Modelled after a typical Home Assistant 2026.4.0 install. No real device
serial numbers, hostnames, or owner data is included.

## Regeneration

These files are version controlled. To add an entity or area:

1. Append the entity dict to `home_default.json`. Keep `entity_id`s
   namespaced by domain (`light.*`, `climate.*`, ...).
2. If introducing a new area, add it to `home_default.areas.json`.
3. If introducing a satellite or other device referenced by a scenario,
   add it to `home_default.devices.json`.
4. Run `python -m pytest container/tests/test_real_scenarios.py -v` to
   confirm nothing regressed.

The corpus is intentionally small (~50 entities) — large enough to
exercise the entity matcher's tie-breaking and area-context paths
without slowing the suite down.
