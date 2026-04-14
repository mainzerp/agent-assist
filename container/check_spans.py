"""Quick script to inspect span timing for the latest trace."""
import sqlite3

conn = sqlite3.connect("/data/agent_assist.db")
c = conn.cursor()
c.execute("SELECT trace_id, user_input FROM trace_summary ORDER BY created_at DESC LIMIT 1")
r = c.fetchone()
print(f"Trace: {r[0]}")
print(f"Input: {r[1][:60]}")
print()

c.execute(
    "SELECT span_name, agent_id, start_time, duration_ms FROM trace_spans WHERE trace_id=? ORDER BY start_time",
    (r[0],),
)
for row in c.fetchall():
    name = row[0] or "?"
    agent = row[1] or ""
    start = (row[2] or "")[11:23]
    dur = row[3] or 0
    print(f"{name:20s} | {agent:15s} | start={start} | dur={dur:>8.1f}ms")

