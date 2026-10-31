"""Database connection session management for SQLite."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
import aiosqlite
from backend.app.core.config import settings


@asynccontextmanager
async def get_db(db_path: Path | None = None) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield an asynchronous SQLite connection configured with WAL and foreign keys."""
    target_path = db_path or settings.DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(target_path) as db:
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA busy_timeout = 5000;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        db.row_factory = aiosqlite.Row
        yield db
