"""Typed repository for Generation Jobs in SQLite."""

from typing import Any
import aiosqlite


class JobRepository:
    """Handles CRUD operations and state transitions for the jobs table."""

    @staticmethod
    async def create(db: aiosqlite.Connection, job_data: dict[str, Any]) -> dict[str, Any]:
        query = """
        INSERT INTO jobs (
            id, prompt, negative_prompt, preset_id, resolution, width, height,
            frame_count, fps, steps, cfg_scale, sampler_name, scheduler, seed,
            status, progress_percent, current_step, total_steps, comfy_prompt_id,
            error_code, error_message, created_at
        ) VALUES (
            :id, :prompt, :negative_prompt, :preset_id, :resolution, :width, :height,
            :frame_count, :fps, :steps, :cfg_scale, :sampler_name, :scheduler, :seed,
            :status, :progress_percent, :current_step, :total_steps, :comfy_prompt_id,
            :error_code, :error_message, datetime('now', 'utc')
        );
        """
        await db.execute(query, job_data)
        await db.commit()
        return await JobRepository.get_by_id(db, job_data["id"])  # type: ignore

    @staticmethod
    async def get_by_id(db: aiosqlite.Connection, job_id: str) -> dict[str, Any] | None:
        query = "SELECT * FROM jobs WHERE id = ?;"
        async with db.execute(query, (job_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def list_jobs(
        db: aiosqlite.Connection,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if status:
            query = "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?;"
            params = (status, limit, offset)
        else:
            query = "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?;"
            params = (limit, offset)  # type: ignore

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    async def claim_next_queued(db: aiosqlite.Connection) -> dict[str, Any] | None:
        """Atomically find the oldest queued job and transition it to preparing."""
        async with db.execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1;"
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            job_id = row["id"]

        await db.execute(
            """
            UPDATE jobs 
            SET status = 'preparing', started_at = datetime('now', 'utc') 
            WHERE id = ? AND status = 'queued';
            """,
            (job_id,),
        )
        await db.commit()
        return await JobRepository.get_by_id(db, job_id)

    @staticmethod
    async def update_status(
        db: aiosqlite.Connection,
        job_id: str,
        status: str,
        comfy_prompt_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        duration_sec: float | None = None,
    ) -> None:
        query = """
        UPDATE jobs
        SET status = :status,
            comfy_prompt_id = COALESCE(:comfy_prompt_id, comfy_prompt_id),
            error_code = :error_code,
            error_message = :error_message,
            duration_sec = COALESCE(:duration_sec, duration_sec),
            completed_at = CASE 
                WHEN :status IN ('completed', 'failed', 'cancelled') THEN datetime('now', 'utc')
                ELSE completed_at
            END
        WHERE id = :job_id;
        """
        await db.execute(
            query,
            {
                "job_id": job_id,
                "status": status,
                "comfy_prompt_id": comfy_prompt_id,
                "error_code": error_code,
                "error_message": error_message,
                "duration_sec": duration_sec,
            },
        )
        await db.commit()

    @staticmethod
    async def update_progress(
        db: aiosqlite.Connection,
        job_id: str,
        progress_percent: float,
        current_step: int,
        total_steps: int,
    ) -> None:
        query = """
        UPDATE jobs
        SET progress_percent = :progress_percent,
            current_step = :current_step,
            total_steps = :total_steps
        WHERE id = :job_id;
        """
        await db.execute(
            query,
            {
                "job_id": job_id,
                "progress_percent": progress_percent,
                "current_step": current_step,
                "total_steps": total_steps,
            },
        )
        await db.commit()

    @staticmethod
    async def cancel(db: aiosqlite.Connection, job_id: str) -> bool:
        """Cancel a job if it is not in a terminal state."""
        query = """
        UPDATE jobs
        SET status = 'cancelled', completed_at = datetime('now', 'utc')
        WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled');
        """
        cursor = await db.execute(query, (job_id,))
        await db.commit()
        return cursor.rowcount > 0

    @staticmethod
    async def reconcile_interrupted(db: aiosqlite.Connection) -> int:
        """Mark any non-terminal jobs from a previous crashed session as failed."""
        query = """
        UPDATE jobs
        SET status = 'failed',
            error_code = 'SERVER_RESTARTED_MID_JOB',
            error_message = 'Job was interrupted by an application or system restart.',
            completed_at = datetime('now', 'utc')
        WHERE status IN ('preparing', 'generating', 'encoding', 'validating');
        """
        cursor = await db.execute(query)
        await db.commit()
        return cursor.rowcount
