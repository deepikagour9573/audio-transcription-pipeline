"""HTTP endpoints for audio transcription."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import services
from app.models import create_job, get_job, update_job
from app.settings import AUDIO_DIR, MAX_BULK_FILES, SUPPORTED_FORMATS

 # All HTTP endpoints live here, so the entry point can stay nice and small.
router = APIRouter()

# A quick health check is useful when starting the app or checking a deployment.
@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": services.model is not None,
        "ffmpeg_available": services.ffmpeg_version is not None,
        "ffmpeg_path": str(services.FFMPEG_EXECUTABLE) if services.FFMPEG_EXECUTABLE else None,
        "ffmpeg_version": services.ffmpeg_version,
    }

# We return 202 because the upload is accepted now, while transcription finishes in the background.
@router.post("/transcriptions", status_code=202)
async def create_transcription(file: UploadFile = File(...)):
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        await file.close()
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Give every upload its own ID so the client can check this job later.
    job_id = str(uuid.uuid4())
    audio_path = AUDIO_DIR / f"{job_id}{suffix}"
    try:
        size = await services.save_upload(file, audio_path)
    finally:
        await file.close()
    if size == 0:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty")

    # Save the queued job before scheduling it, so status checks can find it immediately.
    create_job(job_id, filename, str(audio_path))
    services.schedule_job(job_id)
    return {
        "job_id": job_id, "status": "queued",
        "message": "Audio accepted. Check the job status endpoint for the result.",
    }

# Bulk uploads use the same job flow, but report the result for each file separately.
@router.post("/transcriptions/bulk", status_code=202)
async def create_bulk_transcriptions(
    files: Annotated[list[UploadFile], File(description="Select up to 10 audio files")],
):
    if len(files) > MAX_BULK_FILES:
        for file in files:
            await file.close()
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_BULK_FILES} files can be uploaded at once",
        )

    results = []
    for file in files:
        filename = file.filename or "audio"
        suffix = Path(filename).suffix.lower()
        try:
            if suffix not in SUPPORTED_FORMATS:
                results.append({
                    "filename": filename, "status": "rejected",
                    "error": "Unsupported audio format",
                })
                continue

            job_id = str(uuid.uuid4())
            audio_path = AUDIO_DIR / f"{job_id}{suffix}"
            try:
                size = await services.save_upload(file, audio_path)
            except HTTPException as exc:
                results.append({
                    "filename": filename, "status": "rejected", "error": exc.detail,
                })
                continue

            if size == 0:
                audio_path.unlink(missing_ok=True)
                results.append({
                    "filename": filename, "status": "rejected",
                    "error": "The uploaded audio file is empty",
                })
                continue

            create_job(job_id, filename, str(audio_path))
            services.schedule_job(job_id)
            results.append({"filename": filename, "job_id": job_id, "status": "queued"})
        finally:
            await file.close()

    return {
        "total_files": len(files),
        "accepted": sum(item["status"] == "queued" for item in results),
        "rejected": sum(item["status"] == "rejected" for item in results),
        "results": results,
    }

# The client can poll this endpoint without keeping the original upload request open.
@router.get("/transcriptions/{job_id}")
async def get_transcription(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Transcription job not found")

    response = {
        "job_id": job["job_id"], "filename": job["original_filename"],
        "status": job["status"], "attempts": job["attempts"],
        "created_at": job["created_at"], "updated_at": job["updated_at"],
    }
    if job["status"] == "completed":
        response.update({
            "language": job["language"],
            "duration_seconds": job["duration_seconds"],
            "full_text": job["transcript"],
            "segments": json.loads(job["segments_json"] or "[]"),
        })
    if job["status"] == "failed":
        response["error"] = job["error"]
    return response

# A retry is only useful when the original audio is still available on disk.
@router.post("/transcriptions/{job_id}/retry", status_code=202)
async def retry_transcription(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Transcription job not found")
    if job["status"] != "failed":
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried")
    if not Path(job["audio_path"]).exists():
        raise HTTPException(
            status_code=410,
            detail="Original audio is no longer available, so this job cannot be retried",
        )
    update_job(job_id, status="queued", error=None)
    services.schedule_job(job_id)
    return {"job_id": job_id, "status": "queued", "message": "Retry scheduled"}
