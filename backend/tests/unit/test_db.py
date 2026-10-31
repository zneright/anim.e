"""Comprehensive unit tests for SQLite WAL persistence and schema migrations."""

import uuid
import pytest
import aiosqlite
from backend.app.db.migrations import run_migrations
from backend.app.db.session import get_db
from backend.app.db.repositories.job_repo import JobRepository
from backend.app.db.repositories.output_repo import OutputRepository
from backend.app.db.repositories.setting_repo import SettingRepository
from backend.app.db.repositories.telemetry_repo import TelemetryRepository


@pytest.fixture
async def test_db(tmp_path):
    """Provides an isolated test database with migrations applied."""
    db_file = tmp_path / "test_app.db"
    async with get_db(db_file) as db:
        await run_migrations(db)
        yield db, db_file


@pytest.mark.asyncio
async def test_wal_mode_and_foreign_keys_configured(test_db):
    db, _ = test_db
    async with db.execute("PRAGMA journal_mode;") as cursor:
        row = await cursor.fetchone()
        assert row[0].upper() == "WAL"

    async with db.execute("PRAGMA foreign_keys;") as cursor:
        row = await cursor.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_job_repository_crud_and_transitions(test_db):
    db, _ = test_db
    job_id = str(uuid.uuid4())
    job_payload = {
        "id": job_id,
        "prompt": "1girl, silver hair, anime style",
        "negative_prompt": "lowres, blurry",
        "preset_id": "standard_loop",
        "resolution": "384x384",
        "width": 384,
        "height": 384,
        "frame_count": 16,
        "fps": 8,
        "steps": 16,
        "cfg_scale": 7.0,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "seed": 424242,
        "status": "queued",
        "progress_percent": 0.0,
        "current_step": 0,
        "total_steps": 16,
        "comfy_prompt_id": None,
        "error_code": None,
        "error_message": None,
    }

    # 1. Create Job
    created = await JobRepository.create(db, job_payload)
    assert created["id"] == job_id
    assert created["status"] == "queued"

    # 2. Claim next queued job
    claimed = await JobRepository.claim_next_queued(db)
    assert claimed is not None
    assert claimed["id"] == job_id
    assert claimed["status"] == "preparing"
    assert claimed["started_at"] is not None

    # Next claim should be None because queue is empty
    assert await JobRepository.claim_next_queued(db) is None

    # 3. Update Progress
    await JobRepository.update_progress(db, job_id, 50.0, 8, 16)
    job = await JobRepository.get_by_id(db, job_id)
    assert job["progress_percent"] == 50.0
    assert job["current_step"] == 8

    # 4. Update Status to Completed
    await JobRepository.update_status(db, job_id, "completed", duration_sec=42.5)
    job = await JobRepository.get_by_id(db, job_id)
    assert job["status"] == "completed"
    assert job["duration_sec"] == 42.5
    assert job["completed_at"] is not None


@pytest.mark.asyncio
async def test_output_repository_and_foreign_key_constraint(test_db):
    db, _ = test_db
    job_id = str(uuid.uuid4())
    job_payload = {
        "id": job_id,
        "prompt": "test prompt",
        "negative_prompt": "test neg",
        "preset_id": "fast_draft",
        "resolution": "384x384",
        "width": 384,
        "height": 384,
        "frame_count": 8,
        "fps": 8,
        "steps": 15,
        "cfg_scale": 7.0,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "seed": 12345,
        "status": "completed",
        "progress_percent": 100.0,
        "current_step": 15,
        "total_steps": 15,
        "comfy_prompt_id": "prompt-123",
        "error_code": None,
        "error_message": None,
    }
    await JobRepository.create(db, job_payload)

    # Valid output insertion
    output_id = str(uuid.uuid4())
    out_payload = {
        "id": output_id,
        "job_id": job_id,
        "video_path": "outputs/2026-11/test.mp4",
        "thumbnail_path": "thumbnails/test.webp",
        "file_size_bytes": 102400,
        "width": 384,
        "height": 384,
        "frame_count": 8,
        "fps": 8.0,
        "duration_sec": 1.0,
        "motion_score": 9.18,
        "is_favorited": 0,
    }
    out = await OutputRepository.create(db, out_payload)
    assert out["id"] == output_id
    assert out["prompt"] == "test prompt"

    # Favorite toggle
    assert await OutputRepository.toggle_favorite(db, output_id) is True
    fav_out = await OutputRepository.get_by_id(db, output_id)
    assert fav_out["is_favorited"] == 1

    # Foreign key rejection on invalid job_id
    invalid_out = {
        "id": str(uuid.uuid4()),
        "job_id": "non-existent-job-uuid",
        "video_path": "outputs/invalid.mp4",
        "thumbnail_path": "thumbnails/invalid.webp",
        "file_size_bytes": 1000,
        "width": 384,
        "height": 384,
        "frame_count": 8,
        "fps": 8.0,
        "duration_sec": 1.0,
        "motion_score": 1.0,
        "is_favorited": 0,
    }
    with pytest.raises(aiosqlite.IntegrityError):
        await OutputRepository.create(db, invalid_out)


@pytest.mark.asyncio
async def test_crash_reconciliation(test_db):
    db, _ = test_db
    # Create two jobs that were in-flight during a crash
    for st in ("preparing", "generating"):
        await JobRepository.create(
            db,
            {
                "id": str(uuid.uuid4()),
                "prompt": "test crash",
                "negative_prompt": "",
                "preset_id": "draft",
                "resolution": "384x384",
                "width": 384,
                "height": 384,
                "frame_count": 8,
                "fps": 8,
                "steps": 15,
                "cfg_scale": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "seed": 99,
                "status": st,
                "progress_percent": 30.0,
                "current_step": 5,
                "total_steps": 15,
                "comfy_prompt_id": None,
                "error_code": None,
                "error_message": None,
            },
        )

    reconciled_count = await JobRepository.reconcile_interrupted(db)
    assert reconciled_count == 2

    # Verify status changed to failed
    failed_jobs = await JobRepository.list_jobs(db, status="failed")
    assert len(failed_jobs) == 2
    for j in failed_jobs:
        assert j["error_code"] == "SERVER_RESTARTED_MID_JOB"


@pytest.mark.asyncio
async def test_settings_and_telemetry_repositories(test_db):
    db, _ = test_db
    # Settings
    all_settings = await SettingRepository.get_all(db)
    assert "defaults" in all_settings
    assert "safety" in all_settings
    assert all_settings["safety"]["min_disk_free_gib"] == 15.0

    await SettingRepository.set(db, "theme", {"mode": "dark", "accent": "violet"})
    theme = await SettingRepository.get(db, "theme")
    assert theme["mode"] == "dark"

    # Telemetry
    job_id = str(uuid.uuid4())
    await JobRepository.create(
        db,
        {
            "id": job_id,
            "prompt": "telemetry test",
            "negative_prompt": "",
            "preset_id": "standard_loop",
            "resolution": "384x384",
            "width": 384,
            "height": 384,
            "frame_count": 16,
            "fps": 8,
            "steps": 16,
            "cfg_scale": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "seed": 1,
            "status": "generating",
            "progress_percent": 0.0,
            "current_step": 0,
            "total_steps": 16,
            "comfy_prompt_id": None,
            "error_code": None,
            "error_message": None,
        },
    )

    await TelemetryRepository.record_sample(
        db, job_id, vram_used_mib=3100.0, vram_total_mib=4096.0, gpu_util_pct=75.0, temp_c=65.0, power_w=44.0
    )
    await TelemetryRepository.record_sample(
        db, job_id, vram_used_mib=3340.0, vram_total_mib=4096.0, gpu_util_pct=85.0, temp_c=68.0, power_w=44.9
    )

    summary = await TelemetryRepository.get_job_summary(db, job_id)
    assert summary is not None
    assert summary["sample_count"] == 2
    assert summary["peak_vram_mib"] == 3340.0
    assert summary["avg_gpu_util_pct"] == 80.0
    assert summary["max_temp_c"] == 68.0


@pytest.mark.asyncio
async def test_persistence_across_connection_reopen(test_db):
    db, db_file = test_db
    job_id = str(uuid.uuid4())
    await JobRepository.create(
        db,
        {
            "id": job_id,
            "prompt": "persistence test",
            "negative_prompt": "",
            "preset_id": "fast_draft",
            "resolution": "384x384",
            "width": 384,
            "height": 384,
            "frame_count": 8,
            "fps": 8,
            "steps": 15,
            "cfg_scale": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "seed": 777,
            "status": "queued",
            "progress_percent": 0.0,
            "current_step": 0,
            "total_steps": 15,
            "comfy_prompt_id": None,
            "error_code": None,
            "error_message": None,
        },
    )

    # Re-open connection to same file
    async with get_db(db_file) as new_db:
        job = await JobRepository.get_by_id(new_db, job_id)
        assert job is not None
        assert job["prompt"] == "persistence test"
