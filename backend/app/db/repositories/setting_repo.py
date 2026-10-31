"""Typed repository for Application Settings in SQLite."""

import json
from typing import Any
import aiosqlite


class SettingRepository:
    """Handles storage and retrieval of application-wide JSON settings."""

    @staticmethod
    async def get(db: aiosqlite.Connection, key: str) -> dict[str, Any] | None:
        query = "SELECT value_json FROM settings WHERE key = ?;"
        async with db.execute(query, (key,)) as cursor:
            row = await cursor.fetchone()
            return json.loads(row["value_json"]) if row else None

    @staticmethod
    async def set(db: aiosqlite.Connection, key: str, value: dict[str, Any]) -> None:
        query = """
        INSERT INTO settings (key, value_json, updated_at)
        VALUES (?, ?, datetime('now', 'utc'))
        ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = datetime('now', 'utc');
        """
        await db.execute(query, (key, json.dumps(value)))
        await db.commit()

    @staticmethod
    async def get_all(db: aiosqlite.Connection) -> dict[str, Any]:
        query = "SELECT key, value_json FROM settings;"
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()
            return {row["key"]: json.loads(row["value_json"]) for row in rows}
