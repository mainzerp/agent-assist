# TODO

## Pending Features

- [ ] **Presence Detection in Agents integrieren**: Die Presence-Infrastruktur (PresenceDetector, `get_most_likely_room()`, `context.presence_room`) ist vorhanden. Domain-Agents (light, music, climate, etc.) muessen `task.context.presence_room` als Raum-Kontext nutzen, wenn kein Raum explizit in der Anfrage genannt wird.
- [ ] **HA Entities auf Exposed Entities beschraenken**: Entity-Zugriff auf die in Home Assistant als "exposed" markierten Entities einschraenken, damit nur freigegebene Geraete gesteuert/abgefragt werden koennen.


default location und current time für general agent

climate-agent scheint keine wetter fragen beantwortn zu können





beim starten des container werden embeeding tasks ausgeführt warum?