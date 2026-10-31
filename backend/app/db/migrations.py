"""Forward-only schema migration runner for SQLite."""

from pathlib import Path
import aiosqlite
from backend.app.core.config import settings

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "initial_schema",
        """
        CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            negative_prompt TEXT NOT NULL,
            preset_id TEXT NOT NULL,
            resolution TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            fps INTEGER NOT NULL DEFAULT 8,
            steps INTEGER NOT NULL DEFAULT 16,
            cfg_scale REAL NOT NULL DEFAULT 7.0,
            sampler_name TEXT NOT NULL DEFAULT 'euler_ancestral',
            scheduler TEXT NOT NULL DEFAULT 'normal',
            seed INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('queued', 'preparing', 'generating', 'encoding', 'validating', 'completed', 'failed', 'cancelled')),
            progress_percent REAL NOT NULL DEFAULT 0.0,
            current_step INTEGER NOT NULL DEFAULT 0,
            total_steps INTEGER NOT NULL DEFAULT 16,
            comfy_prompt_id TEXT,
            error_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
            started_at TEXT,
            completed_at TEXT,
            duration_sec REAL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);

        CREATE TABLE IF NOT EXISTS outputs (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL UNIQUE,
            video_path TEXT NOT NULL,
            thumbnail_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            fps REAL NOT NULL,
            duration_sec REAL NOT NULL,
            motion_score REAL,
            is_favorited INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_outputs_created ON outputs(created_at DESC);

        CREATE TABLE IF NOT EXISTS hardware_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            sample_time TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
            vram_used_mib REAL NOT NULL,
            vram_total_mib REAL NOT NULL DEFAULT 4096.0,
            gpu_util_pct REAL NOT NULL,
            temp_c REAL NOT NULL,
            power_w REAL,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_telemetry_job ON hardware_telemetry(job_id);

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
        );

        INSERT OR IGNORE INTO settings (key, value_json) VALUES 
        ('defaults', '{"preset": "standard_loop", "resolution": "384x384", "frames": 16, "fps": 8, "steps": 16, "cfg": 7.0}'),
        ('safety', '{"min_disk_free_gib": 15.0, "max_vram_limit_mib": 3600.0, "step_timeout_sec": 90.0}');
        """,
    )
]


async def run_migrations(db: aiosqlite.Connection) -> list[int]:
    """Execute all pending forward migrations safely within transactions."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
        );
        """
    )
    await db.commit()

    applied_versions: set[int] = set()
    async with db.execute("SELECT version FROM schema_versions ORDER BY version ASC;") as cursor:
        rows = await cursor.fetchall()
        for row in rows:
            applied_versions.add(row[0])

    newly_applied: list[int] = []
    for version, name, script in MIGRATIONS:
        if version not in applied_versions:
            # Execute within transaction
            await db.executescript(script)
            await db.execute(
                "INSERT INTO schema_versions (version, name) VALUES (?, ?);",
                (version, name),
            )
            await db.commit()
            newly_applied.append(version)

    return newly_applied


async def init_db(db_path: Path | None = None) -> list[int]:
    """Convenience entrypoint to initialize database and apply pending migrations."""
    from backend.app.db.session import get_db

    async with get_db(db_path) as db:
        return await run_migrations(db)
