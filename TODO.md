# TODO

## Pending Features

- [ ] **HA Entities auf Exposed Entities beschraenken**: Entity-Zugriff auf die in Home Assistant als "exposed" markierten Entities einschraenken, damit nur freigegebene Geraete gesteuert/abgefragt werden koennen.

- [x] **Entity-Historie / Recorder**: Zugriff auf Home-Assistant-Recorder-Zeitreihen (z. B. "Wie warm war es gestern?"), analog zu einem dedizierten History-Tool

- [ ] **Nutzer- und Agent-Memory**: Persistente Profile, Memory-Tool (speichern/abrufen/aktualisieren), Limits/Eviction, optional UI im Dashboard; Mehrschichtige Nutzerzuordnung wo sinnvoll.

- [x] **Cancel-Intent / Dismiss**: Orchestrator-LLM routet zu virtuellem Agent **cancel-interaction**; der Container liefert einen kurzen ACK **ohne** Domain-Dispatch (Manifest **0.5.5**). HA-Integration leitet den User-Text immer an den Container weiter (kein lokales Keyword-Shortcut).

- [ ] **HA-Service fuer Automationen (`ai_task`-Aequivalent)**: Service oder klarer Contract fuer Automatisierungen (z. B. strukturierter Output / `generate_data`-Pattern), der den Container ohne manuelles HTTP-Basteln nutzbar macht.

- [ ] **Kalender: lesen, proaktive Reminder, Zuordnungen**: Kalenderereignisse lesen, gestufte/proaktive Erinnerungen, optional Nutzer-zu-Kalender-Mappings (wie im Smart-Assist-Prompt-Pattern); Dashboard/Traces wo passend.

- [x] **Einheitliches Control-Tool und parallele Ausfuehrung**: **Umgesetzt (Parallelitaet):** (1) Orchestrator dispatcht mehrere unabhaengige Intents parallel an Domain-Agenten (`asyncio.gather`). (2) `complete_with_tools` fuehrt mehrere `tool_calls` einer LLM-Runde parallel (`asyncio.gather`, **0.18.18**); Tool-Messages bleiben in ``tool_calls``-Reihenfolge. **Bewusst nicht:** ein einziges generisches HA-**Batch**-Tool (eine Function-Invocation mit vielen HA-Aktionen); stattdessen Mehrfach-Steuerung ueber Orchestrator-Mehrzeilen bzw. kuenftig optional gebundelte JSON-Actions im Domain-Agent.


wecker:
standard einmalig, wiederkehrend konfigurierbar.

wecker:
wecken mit infos anreichern, wetter, news usw. (bereistellung der daten durch div. agents)

beisp.

Guten Morgen! Heite ist Sonntag der 26. April 2026
draußen ist, Strahlender Sonnenschein, aktuell 19 °C – heute bleibt es trocken und schön. Die Woche startet ähnlich mild, perfektes Frühlingswetter!
Kurze News
Syrien: Ein ehemaliger General der Assad-Armee steht vor Gericht – er muss sich wegen Verbrechen gegen das syrische Volk verantworten.
Nahost: Im Korruptionsverfahren gegen Israels Premier Netanjahu spricht sich Staatspräsident Herzog für eine außergerichtliche Einigung aus – Netanjahu lehnt dies bislang ab.
Iran: Trotz der seit dem 8. April geltenden Feuerpause hat sich die humanitäre Lage im Iran laut UN weiter verschlechtert.
USA: Beim White House Correspondents' Dinner in Washington gab es einen Zwischenfall – ein Verdächtiger eröffnete das Feuer und verletzte einen Secret-Service-Agenten, bevor er gestoppt wurde.
Jahrestag: Heute vor 40 Jahren ereignete sich die Katastrophe von Tschernobyl – ein Datum, das Deutschlands Energiepolitik bis heute prägt.
Schönen Sonntag!