"""Typed repository for GPU Telemetry Samples in SQLite."""

from typing import Any
import aiosqlite


class TelemetryRepository:
    """Handles logging and aggregation of hardware telemetry samples."""

    @staticmethod
    async def record_sample(
        db: aiosqlite.Connection,
        job_id: str,
        vram_used_mib: float,
        vram_total_mib: float,
        gpu_util_pct: float,
        temp_c: float,
        power_w: float | None = None,
    ) -> None:
        query = """
        INSERT INTO hardware_telemetry (
            job_id, sample_time, vram_used_mib, vram_total_mib,
            gpu_util_pct, temp_c, power_w
        ) VALUES (?, datetime('now', 'utc'), ?, ?, ?, ?, ?);
        """
        await db.execute(
            query,
            (job_id, vram_used_mib, vram_total_mib, gpu_util_pct, temp_c, power_w),
        )
        await db.commit()

    @staticmethod
    async def get_job_summary(db: aiosqlite.Connection, job_id: str) -> dict[str, Any] | None:
        query = """
        SELECT 
            COUNT(*) as sample_count,
            MAX(vram_used_mib) as peak_vram_mib,
            AVG(gpu_util_pct) as avg_gpu_util_pct,
            MAX(gpu_util_pct) as peak_gpu_util_pct,
            MAX(temp_c) as max_temp_c,
            MAX(power_w) as max_power_w
        FROM hardware_telemetry
        WHERE job_id = ?;
        """
        async with db.execute(query, (job_id,)) as cursor:
            row = await cursor.fetchone()
            if not row or row["sample_count"] == 0:
                return None
            return {
                "sample_count": row["sample_count"],
                "peak_vram_mib": round(row["peak_vram_mib"], 1),
                "avg_gpu_util_pct": round(row["avg_gpu_util_pct"], 1),
                "peak_gpu_util_pct": round(row["peak_gpu_util_pct"], 1),
                "max_temp_c": round(row["max_temp_c"], 1),
                "max_power_w": round(row["max_power_w"], 1) if row["max_power_w"] else None,
            }
