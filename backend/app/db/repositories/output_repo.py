"""Typed repository for Generated Video Outputs in SQLite."""

from typing import Any
import aiosqlite


class OutputRepository:
    """Handles CRUD operations for generated video assets and gallery items."""

    @staticmethod
    async def create(db: aiosqlite.Connection, output_data: dict[str, Any]) -> dict[str, Any]:
        query = """
        INSERT INTO outputs (
            id, job_id, video_path, thumbnail_path, file_size_bytes,
            width, height, frame_count, fps, duration_sec, motion_score,
            is_favorited, created_at
        ) VALUES (
            :id, :job_id, :video_path, :thumbnail_path, :file_size_bytes,
            :width, :height, :frame_count, :fps, :duration_sec, :motion_score,
            COALESCE(:is_favorited, 0), datetime('now', 'utc')
        );
        """
        await db.execute(query, output_data)
        await db.commit()
        return await OutputRepository.get_by_id(db, output_data["id"])  # type: ignore

    @staticmethod
    async def get_by_id(db: aiosqlite.Connection, output_id: str) -> dict[str, Any] | None:
        query = """
        SELECT o.*, j.prompt, j.negative_prompt, j.preset_id, j.seed, j.steps, j.cfg_scale
        FROM outputs o
        JOIN jobs j ON o.job_id = j.id
        WHERE o.id = ?;
        """
        async with db.execute(query, (output_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def get_by_job_id(db: aiosqlite.Connection, job_id: str) -> dict[str, Any] | None:
        query = "SELECT * FROM outputs WHERE job_id = ?;"
        async with db.execute(query, (job_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def list_outputs(
        db: aiosqlite.Connection,
        favorites_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        base_query = """
        SELECT o.*, j.prompt, j.preset_id, j.seed
        FROM outputs o
        JOIN jobs j ON o.job_id = j.id
        """
        if favorites_only:
            query = f"{base_query} WHERE o.is_favorited = 1 ORDER BY o.created_at DESC LIMIT ? OFFSET ?;"
        else:
            query = f"{base_query} ORDER BY o.created_at DESC LIMIT ? OFFSET ?;"

        async with db.execute(query, (limit, offset)) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    async def toggle_favorite(db: aiosqlite.Connection, output_id: str) -> bool:
        query = """
        UPDATE outputs 
        SET is_favorited = CASE WHEN is_favorited = 1 THEN 0 ELSE 1 END
        WHERE id = ?;
        """
        cursor = await db.execute(query, (output_id,))
        await db.commit()
        return cursor.rowcount > 0

    @staticmethod
    async def delete(db: aiosqlite.Connection, output_id: str) -> bool:
        query = "DELETE FROM outputs WHERE id = ?;"
        cursor = await db.execute(query, (output_id,))
        await db.commit()
        return cursor.rowcount > 0
