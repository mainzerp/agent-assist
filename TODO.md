# TODO

## Pending Features

- [ ] **Presence Detection in Agents integrieren**: Die Presence-Infrastruktur (PresenceDetector, `get_most_likely_room()`, `context.presence_room`) ist vorhanden. Domain-Agents (light, music, climate, etc.) muessen `task.context.presence_room` als Raum-Kontext nutzen, wenn kein Raum explizit in der Anfrage genannt wird.
- [ ] **HA Entities auf Exposed Entities beschraenken**: Entity-Zugriff auf die in Home Assistant als "exposed" markierten Entities einschraenken, damit nur freigegebene Geraete gesteuert/abgefragt werden koennen.

- [x] **Entity-Historie / Recorder**: Zugriff auf Home-Assistant-Recorder-Zeitreihen (z. B. "Wie warm war es gestern?"), analog zu einem dedizierten History-Tool

- [ ] **Nutzer- und Agent-Memory**: Persistente Profile, Memory-Tool (speichern/abrufen/aktualisieren), Limits/Eviction, optional UI im Dashboard; Mehrschichtige Nutzerzuordnung wo sinnvoll.

- [x] **Cancel-Intent / Dismiss**: Orchestrator-LLM routet zu virtuellem Agent **cancel-interaction**; der Container liefert einen kurzen ACK **ohne** Domain-Dispatch (Manifest **0.5.5**). HA-Integration leitet den User-Text immer an den Container weiter (kein lokales Keyword-Shortcut).

- [ ] **HA-Service fuer Automationen (`ai_task`-Aequivalent)**: Service oder klarer Contract fuer Automatisierungen (z. B. strukturierter Output / `generate_data`-Pattern), der den Container ohne manuelles HTTP-Basteln nutzbar macht.

- [ ] **Kalender: lesen, proaktive Reminder, Zuordnungen**: Kalenderereignisse lesen, gestufte/proaktive Erinnerungen, optional Nutzer-zu-Kalender-Mappings (wie im Smart-Assist-Prompt-Pattern); Dashboard/Traces wo passend.

- [x] **Einheitliches Control-Tool und parallele Ausfuehrung**: **Umgesetzt (Parallelitaet):** (1) Orchestrator dispatcht mehrere unabhaengige Intents parallel an Domain-Agenten (`asyncio.gather`). (2) `complete_with_tools` fuehrt mehrere `tool_calls` einer LLM-Runde parallel (`asyncio.gather`, **0.18.18**); Tool-Messages bleiben in ``tool_calls``-Reihenfolge. **Bewusst nicht:** ein einziges generisches HA-**Batch**-Tool (eine Function-Invocation mit vielen HA-Aktionen); stattdessen Mehrfach-Steuerung ueber Orchestrator-Mehrzeilen bzw. kuenftig optional gebundelte JSON-Actions im Domain-Agent.