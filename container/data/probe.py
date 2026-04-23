import asyncio, json
from app.cache.vector_store import COLLECTION_ENTITY_INDEX, get_vector_store
from app.entity.index import EntityIndex
from app.entity.matcher import EntityMatcher

async def main():
    vs = await get_vector_store()
    print("vector entries:", vs.count(COLLECTION_ENTITY_INDEX))
    idx = EntityIndex(vector_store=vs)
    matcher = EntityMatcher(entity_index=idx)
    queries = ["schlafzimmer", "bedroom", "wohnzimmer", "living room", "masterbad", "bad", "bathroom", "chambre", "dormitorio"]
    for q in queries:
        try:
            res = matcher.match(q)
            if asyncio.iscoroutine(res):
                res = await res
        except Exception as e:
            print(q, "ERROR:", e); continue
        cands = getattr(res, "candidates", None)
        if cands is None and isinstance(res, list): cands = res
        if cands is None: 
            print(q, "=>", res); continue
        top = cands[:3]
        out = []
        for c in top:
            eid = getattr(c, "entity_id", None) or (c.get("entity_id") if isinstance(c, dict) else None)
            sc = getattr(c, "score", None)
            if sc is None and isinstance(c, dict): sc = c.get("score", 0)
            out.append((eid, round(float(sc or 0), 3)))
        print(q, "=>", out)
asyncio.run(main())
