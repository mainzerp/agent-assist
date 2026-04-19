import asyncio

import aiosqlite


async def main():
    db = await aiosqlite.connect("/data/agent_assist.db")
    rows = await (await db.execute("SELECT agent_id, model, max_tokens, temperature FROM agent_configs")).fetchall()
    for _r in rows:
        pass
    await db.close()


asyncio.run(main())
