# TODO

## Pending Features

- [ ] **Presence Detection in Agents integrieren**: Die Presence-Infrastruktur (PresenceDetector, `get_most_likely_room()`, `context.presence_room`) ist vorhanden. Domain-Agents (light, music, climate, etc.) muessen `task.context.presence_room` als Raum-Kontext nutzen, wenn kein Raum explizit in der Anfrage genannt wird.
- [ ] **HA Entities auf Exposed Entities beschraenken**: Entity-Zugriff auf die in Home Assistant als "exposed" markierten Entities einschraenken, damit nur freigegebene Geraete gesteuert/abgefragt werden koennen.


Send-Agent

user fragt nach einem rezept und will das es auf sein smartphone (z.b. lauras handy) gesendet wird

orchestrator lässt general-agent nach rezept und link suchen
orchestrator nutzt send agent um die infos an ein smartphone zu schicken (notify über companion app), oder via tts auf einen satellite

nben dem agent wird eine ui page benötigt die alle notify devices anzeigt um ein mapping zu einem benutzer namen machen zu können. 