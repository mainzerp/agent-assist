"""Schema versioning and migrations.

Tracks the current schema version and applies migrations sequentially
to bring the database up to the latest version.
"""

import aiosqlite


CURRENT_VERSION = 1


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Get the current schema version from the database."""
    try:
        cursor = await db.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else 0
    except aiosqlite.OperationalError:
        return 0


async def set_schema_version(db: aiosqlite.Connection, version: int) -> None:
    """Record a new schema version."""
    await db.execute(
        "INSERT INTO schema_version (version) VALUES (?)", (version,)
    )


async def run_migrations(db: aiosqlite.Connection) -> None:
    """Run all pending migrations sequentially.

    Called at startup after init_db(). Each migration function
    handles one version increment.
    """
    current = await get_schema_version(db)

    migrations = {
        # Version 1 is the initial schema created by init_db().
        # Future migrations go here:
        # 2: _migrate_v2,
        # 3: _migrate_v3,
    }

    for version in sorted(migrations.keys()):
        if version > current:
            await migrations[version](db)
            await set_schema_version(db, version)

    await db.commit()
